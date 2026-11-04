"""Tensor decomposition search for 3x3 * 3x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations
import sys

DIMENSIONS = [3, 3, 7]

# Deadline guard: return best result before 150s wall-clock
DEADLINE = 150
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME) - 5  # 5s safety margin

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] if matmul identity holds.
    Actually the tensor T[i,j,k] = delta... let me reconsider.
    
    The multiplication tensor T has entries T[i,j,k] = 1 if the (i,k) entry of A*B equals sum_j A[i,j]*B[j,k].
    So T[i,j,k] = 1 for all valid (i,j,k) in the standard basis.
    
    Wait, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T is in U*V*W* space where:
    - First index: (i, j) for A -> dimension n*m
    - Second index: (j, k) for B -> dimension m*p  
    - Third index: (i, k) for C -> dimension p*n (trace convention W[k*n+i])
    
    T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k' else 0
    
    So T is the identity-like tensor. For a rank-r decomposition:
    T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    Meaning: for each (i,j), (j',k), (i',k'):
    sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = T[(i,j),(j',k),(i',k')]
    = 1 if i=i', j=j', k=k' else 0
    """
    n, m, p = n, m, p
    rank = len(U[0]) if U else 0
    # Build the tensor T as a 3D array: T[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,p*n)
    # T[a,b,c] = 1 if a//m == c%n (i==i'), a%m == b//p (j==j'), b%p == c//n (k==k')
    # Actually: a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 iff i==i', j==j', k==k'
    # i = a//m, j = a%m, j' = b//p, k = b%p, i' = c%n, k' = c//n
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # T[a,b,c] should be 1
                val = 0
                for cc in range(rank):
                    val += U[a][cc] * V[b][cc] * W[c][cc]
                if val != 1:
                    return False
                # Check that off-diagonal entries are 0
                # This is expensive; do a sample check
    # Sample some off-diagonal entries
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i, j = divmod(a, m)
        jp, k = divmod(b, p)
        kp, ip = divmod(c, n)
        if i == ip and j == jp and k == kp:
            continue  # skip diagonal
        val = 0
        for cc in range(rank):
            val += U[a][cc] * V[b][cc] * W[c][cc]
        if val != 0:
            return False
    return True

def verify_full(U, V, W, n, m, p):
    """Full verification - expensive but thorough."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][cc] * V[b][cc] * W[c][cc] for cc in range(rank))
                if val != 1:
                    return False
    # Check all off-diagonal
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i, j = divmod(a, m)
                jp, k = divmod(b, p)
                kp, ip = divmod(c, n)
                if i == ip and j == jp and k == kp:
                    continue
                val = sum(U[a][cc] * V[b][cc] * W[c][cc] for cc in range(rank))
                if val != 0:
                    return False
    return True

def schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def solve_linear_system(A, b):
    """Solve Ax = b over rationals using numpy with exact fraction conversion.
    A: list of lists, b: list. Returns solution or None."""
    A_arr = np.array(A, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    try:
        x = np.linalg.solve(A_arr, b_arr)
        # Check if solution is close to half-integers
        x2 = x * 2
        if np.allclose(x2, np.round(x2), atol=1e-8):
            return [Fraction(int(round(v))).limit_denominator(1000000) for v in x]
        # Try to find rational approximation
        result = []
        for v in x:
            frac = Fraction(v).limit_denominator(1000000)
            result.append(frac)
        return result
    except np.linalg.LinAlgError:
        return None

def try_rank_with_support(n, m, p, rank, support_pattern=None, timeout=30):
    """Try to find a rank-`rank` decomposition.
    If support_pattern is given, it's a set of (a,b,c) triples where the tensor has 1s.
    We need: for each (a,b,c) in support: sum_d U[a,d]*V[b,d]*W[c,d] = 1
    And for all other (a,b,c): sum_d U[a,d]*V[b,d]*W[c,d] = 0
    
    This is a system of equations in the variables U[a,d], V[b,d], W[c,d].
    It's bilinear, so we fix two matrices and solve for the third.
    """
    start = time.time()
    
    if support_pattern is None:
        # Standard support: all (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
        support = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    support.add((a, b, c))
    else:
        support = set(support_pattern)
    
    # Strategy: Fix U and V to be random sparse matrices, solve for W.
    # Or use a structured approach.
    
    # Approach: Use the fact that for rank r, we can write:
    # T = sum_{d=0}^{r-1} u_d ⊗ v_d ⊗ w_d
    # where u_d is in R^{n*m}, v_d in R^{m*p}, w_d in R^{p*n}
    
    # The constraint is that T is the "identity" tensor for matmul.
    # T[a,b,c] = 1 iff a//m == c%n, a%m == b//p, b%p == c//n
    
    # Let's try: fix U to have a specific structure.
    # For each column d, U[:,d] should be a vector in R^{n*m}.
    
    # Simple approach: try small integer/rational values.
    # For rank r, we have r*(n*m + m*p + p*n) unknowns.
    # Constraints: n*m*p equations (one per (i,j,k) triple that must equal 1)
    # plus (n*m)*(m*p)*(p*n) - n*m*p equations that must equal 0.
    # That's way too many constraints for large dimensions.
    
    # Better: Use the structure. The tensor T satisfies:
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # For a rank-r decomposition with factors U (n*m x r), V (m*p x r), W (p*n x r):
    # sum_d U[i*m+j, d] * V[j'*p+k, d] * W[k'*n+i', d] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # This means for fixed (i,j,k): sum_d U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d] = 1
    # And for all other (i',j',k'): sum_d U[i*m+j,d] * V[j'*p+k',d] * W[k'*n+i',d] = 0
    
    # Let's try a structured approach:
    # Assume U[i*m+j, d] = u_i(d) * v_j(d) (rank-1 in the i-j block)
    # Then: sum_d u_i(d)*v_j(d) * V[j'*p+k,d] * W[k'*n+i',d] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    
    # This is getting complex. Let's try a direct search for small ranks.
    
    # For now, return schoolbook if we can't find better.
    if time.time() - start > timeout:
        return None
    
    return None

def try_structured_decomposition(n, m, p, max_rank=None, timeout=60):
    """Try to find a lower-rank decomposition using structured patterns."""
    start = time.time()
    best = schoolbook(n, m, p)
    best_rank = n * m * p
    
    if max_rank is None:
        max_rank = best_rank - 1
    
    # Try ranks from best_rank-1 downward
    # For each rank, try to find a valid decomposition
    
    # Key insight: The matmul tensor T_{n,m,p} has known rank bounds.
    # For 3x3 * 3x7:
    #   Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    #   Also: n*m*p / min(n,m,p) = 63/3 = 21
    #   Upper bound: n*m*p = 63 (schoolbook)
    #   Known: rank of 3x3*3x3 is at most 23 (Strassen-like)
    #   For 3x3*3x7, we can potentially do better than 63.
    
    # Strategy: Use the block structure.
    # A (3x3) * B (3x7) = C (3x7)
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # We can group the 7 columns of B.
    # Let's try to find a decomposition with rank < 63.
    
    # Approach: Fix U and V to be random, solve for W, check consistency.
    # This is a system of quadratic equations, so we use iterative methods.
    
    # Alternative: Use the fact that the tensor is a "product" structure.
    # T = sum over j of (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_i ⊗ e_k)
    # = sum_j (sum_i e_i ⊗ e_j) ⊗ (e_j ⊗ I) ⊗ (sum_k e_k ⊗ e_i) ... no this doesn't work directly.
    
    # Let's try a direct numerical approach:
    # For a given rank r, set up the equations and try to solve.
    
    # The equations are:
    # For each (i,j,k) in [0,n)x[0,m)x[0,p):
    #   sum_{d=0}^{r-1} U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d] = 1
    # For each (i,j,k,i',j',k') not all matching:
    #   sum_{d=0}^{r-1} U[i*m+j, d] * V[j'*p+k, d] * W[k'*n+i', d] = 0
    
    # This is too many equations. Instead, we can use:
    # Fix U and V, then W is determined by the first set of equations (if consistent).
    
    # Let's try: Set U and V to random matrices with entries in {0, 1, -1},
    # then solve for W from the "on-support" equations, and verify the "off-support" ones.
    
    import random
    random.seed(12345)
    
    for r in range(min(max_rank, best_rank - 1), 20, -1):  # Try from high to low, stop at 21
        if time.time() - start > timeout:
            break
        if time_remaining() < 10:
            break
        
        found = False
        for trial in range(200):
            if time.time() - start > timeout or time_remaining() < 5:
                break
            
            # Random U and V with small integer entries
            U = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(m * p)]
            
            # Solve for W from the on-support equations:
            # For each (i,j,k): sum_d U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d] = 1
            # This gives n*m*p equations in p*n*r unknowns (W entries).
            # But W[k*n+i, d] only appears in equation (i,j,k) for fixed i,k and varying j.
            # Actually, W[c, d] where c = k*n+i appears in equations for all j.
            # So for fixed (i,k) and d: W[k*n+i, d] appears in equations (i,j,k) for j=0..m-1.
            
            # Let's set up the linear system for W.
            # Unknowns: W[c, d] for c in [0, p*n), d in [0, r) -> p*n*r unknowns
            # Equations: for each (i,j,k): sum_d U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d] = 1
            # That's n*m*p equations.
            
            # The system is: A * w = b where w is the vectorized W.
            # Row for (i,j,k): has entries U[i*m+j,d]*V[j*p+k,d] at position (k*n+i)*r + d
            
            # Build the system
            num_eq = n * m * p
            num_unk = p * n * r
            A_mat = [[0.0] * num_unk for _ in range(num_eq)]
            b_vec = [0.0] * num_eq
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        for d in range(r):
                            A_mat[eq_idx][c * r + d] = U[a][d] * V[b][d]
                        b_vec[eq_idx] = 1.0
                        eq_idx += 1
            
            # Solve A_mat * w = b_vec
            try:
                A_np = np.array(A_mat)
                b_np = np.array(b_vec)
                # Use least squares since the system may be underdetermined
                w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                
                # Check if the solution is close to integer/half-integer
                w_check = w_sol * 2
                if not np.allclose(w_check, np.round(w_check), atol=1e-6):
                    continue
                
                # Convert to fractions
                W_sol = [[0] * r for _ in range(p * n)]
                for c in range(p * n):
                    for d in range(r):
                        val = w_sol[c * r + d]
                        W_sol[c][d] = Fraction(int(round(val * 2))) / 2
                
                # Verify full decomposition
                if verify_full(U, V, W_sol, n, m, p):
                    found = True
                    best = {"U": U, "V": V, "W": W_sol}
                    best_rank = r
                    break
                    
            except Exception:
                continue
        
        if found:
            break
    
    return best, best_rank

def try_gf2_support(n, m, p, timeout=30):
    """Try to find a sparse support using GF(2) linear algebra."""
    start = time.time()
    
    # Over GF(2), the tensor T has entries 1 at positions (i,j,k) mapped to (a,b,c).
    # We want to find a rank-r decomposition over GF(2).
    # This means finding matrices U, V, W over GF(2) such that
    # sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c] (mod 2)
    
    # For the standard matmul tensor, T[a,b,c] = 1 iff i=i', j=j', k=k'.
    
    # The rank over GF(2) is at most the rank over the reals.
    # Let's try to find the minimum rank over GF(2).
    
    # For small ranks, we can try all combinations of support patterns.
    # But that's exponential. Instead, use a greedy approach.
    
    # Actually, let's just try to verify if rank 21 or 22 is achievable.
    # The theoretical lower bound is 21.
    
    # For now, this function is a placeholder for future work.
    if time.time() - start > timeout:
        return None
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better decomposition
    if time_remaining() > 30:
        try:
            result, rank = try_structured_decomposition(n, m, p, max_rank=best_rank - 1, timeout=60)
            if result is not None and rank < best_rank:
                best = result
                best_rank = rank
        except Exception as e:
            pass
    
    # Try GF(2) approach
    if time_remaining() > 15:
        try:
            gf2_result = try_gf2_support(n, m, p, timeout=20)
            if gf2_result is not None:
                # Verify and update
                pass
        except Exception as e:
            pass
    
    return best

# Keep the entry point exactly as required
def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better decomposition with structured approach
    if time_remaining() > 30:
        try:
            result, rank = try_structured_decomposition(n, m, p, max_rank=best_rank - 1, timeout=60)
            if result is not None and rank < best_rank:
                best = result
                best_rank = rank
        except Exception:
            pass
    
    return best
