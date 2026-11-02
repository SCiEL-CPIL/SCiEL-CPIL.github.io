"""Search for low-rank exact tensor decomposition of 2x2 by 2x6 matrix multiplication."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Tensor T[i,j,k] = 1 if i,j,k are valid indices
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    # Also check that "impossible" combinations give 0
    # In our setup, all (i,j,k) in [0,n)x[0,m)x[0,p) should give 1
    return True

def _try_rank_decomposition(n, m, p, rank, time_limit=10.0):
    """
    Try to find a rank-`rank` decomposition.
    
    Strategy: Use the structure that U, V, W have specific sparsity patterns.
    For each term c, U[c] picks a (i,j) pair, V[c] picks a (j,k) pair, W[c] picks a (k,i) pair.
    The constraint is: for each (i,j,k), sum over c of [U[c] selects (i,j)] * [V[c] selects (j,k)] * [W[c] selects (k,i)] = 1
    And for "wrong" combinations, the sum should be 0.
    
    Actually, the tensor is: T[i,j,k] = 1 for all valid (i,j,k).
    So we need: for all (i,j,k) in [0,n)x[0,m)x[0,p): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    This is a system of polynomial equations. Let's try a different approach:
    Fix the support patterns (which (i,j) each U term uses, etc.) and solve for coefficients.
    """
    start = time.time()
    
    # n=2, m=2, p=6
    # U has 2*2=4 rows, V has 2*6=12 rows, W has 6*2=12 rows
    # Each column c of U is a vector in R^4, V in R^12, W in R^12
    
    # Key insight: We can enumerate the "type" of each term.
    # Term c has: u_c in R^4, v_c in R^12, w_c in R^12
    # Constraint: for all (i,j,k): sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
    # And these are the only constraints (24 equations).
    
    # For rank r, we have 3*r*(dim) unknowns but 24 equations.
    # For r >= 8, we have more unknowns than equations.
    
    # Let's try a structured approach: 
    # Assume each term c has u_c supported on exactly one (i,j), v_c on one (j,k), w_c on one (k,i).
    # Then the constraint becomes: for each (i,j,k), the number of terms with that exact (i,j,k) triple = 1,
    # and no term should contribute to a "wrong" (i',j',k') where the triple doesn't match.
    
    # Actually with single-support, a term (i,j),(j,k),(k,i) only contributes to (i,j,k).
    # So we'd need exactly one term for each of the 24 (i,j,k) triples -> rank 24.
    
    # To get lower rank, we need terms with multi-support.
    
    # Let's try: enumerate all possible support patterns for a small number of terms
    # and solve the resulting system.
    
    # Alternative: use the fact that this is a known problem. 
    # mu(2,2,6) - let's try to construct it.
    
    # A 2x2 * 2x6 = 2x6 matrix.
    # C = A*B where A is 2x2, B is 2x6.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Rank-1 terms: (a_i * b_j) * (c_k) would give outer product structure.
    # For term r: C[i,k] += (sum_j U[i,j,r] * V[j,k,r]) * W[k,i,r]
    
    # Let me try to use a known construction.
    # For 2x2x6, we can use the 2x2x2 decomposition (rank 7) for each of the 3 blocks?
    # But 7*3 = 21, not great.
    
    # Let's try a random/optimization approach for rank r from 14 upward.
    
    best_result = None
    best_rank = None
    
    for r in range(max(14, n*m*p // 2), n*m*p + 1):
        if time.time() - start > time_limit:
            break
        
        # Try to find a decomposition of rank r
        # Use random initialization + Newton or just check if feasible
        success = False
        for attempt in range(50):
            if time.time() - start > time_limit:
                break
            
            np.random.seed(attempt * 12345 + r * 678)
            
            # Random initialization
            U_r = np.random.randn(n*m, r) * 0.1
            V_r = np.random.randn(m*p, r) * 0.1
            W_r = np.random.randn(p*n, r) * 0.1
            
            # Normalize columns to help convergence
            for c in range(r):
                un = np.linalg.norm(U_r[:, c])
                vn = np.linalg.norm(V_r[:, c])
                wn = np.linalg.norm(W_r[:, c])
                if un > 1e-10:
                    U_r[:, c] /= un
                if vn > 1e-10:
                    V_r[:, c] /= vn
                if wn > 1e-10:
                    W_r[:, c] /= wn
            
            # Set up the system: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is nonlinear. Let's use a different approach.
            
            # Fix U and V, then for each (i,j,k), W[k*n+i,c] appears linearly across k and i.
            # Actually, let's fix U and V and solve for W.
            # For fixed U, V: the constraint on W is:
            # For each (i,k): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
            # Let M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            # Then: sum_c W[k*n+i,c] * M[i,k,c] = 1 for all (i,k)
            
            # This gives us a linear system for W if we fix U, V!
            # W has p*n*r = 12*r unknowns, and n*p = 12 equations.
            # For r >= 1, we have more unknowns than equations, so a solution exists if the system is consistent.
            
            # Let's build the matrix: for each (i,k) equation, and each (c) term in W[k*n+i,c]
            # The coefficient of W[k*n+i,c] in equation (i,k) is M[i,k,c]
            # But W[k*n+i,c] only appears in equation (i,k) for that specific (i,k).
            
            # So the system is block-diagonal! For each (i,k), we have:
            # sum_c M[i,k,c] * W[k*n+i,c] = 1
            # This is r unknowns for each (i,k), with 1 equation.
            # As long as M[i,k,:] is not all zeros, we can find W[k*n+i,:] satisfying this.
            
            # So the strategy is:
            # 1. Random U, V
            # 2. For each (i,k), compute M[i,k,:] = sum_j U[i*m+j,:] * V[j*p+k,:]
            # 3. Check that M[i,k,:] is not all zero (so we can find W)
            # 4. Set W[k*n+i,:] to any solution of M[i,k,:] @ W[k*n+i,:] = 1
            
            # But we also need the entries to be "nice" (integers/half-integers ideally).
            # For now, let's just find any real solution.
            
            M = np.zeros((n, p, r))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        M[i, k, :] += U_r[i*m+j, :] * V_r[j*p+k, :]
            
            # Check feasibility
            feasible = True
            for i in range(n):
                for k in range(p):
                    if np.all(np.abs(M[i, k, :]) < 1e-10):
                        feasible = False
                        break
                if not feasible:
                    break
            
            if not feasible:
                continue
            
            # For each (i,k), find W[k*n+i,:] such that M[i,k,:] @ W[k*n+i,:] = 1
            W_r = np.zeros((p*n, r))
            valid = True
            for i in range(n):
                for k in range(p):
                    row = M[i, k, :]
                    # Solve row @ x = 1 (underdetermined if r > 1)
                    # Use minimum norm solution
                    if np.linalg.norm(row) < 1e-10:
                        valid = False
                        break
                    x = row / (row @ row)  # Minimum norm solution to row @ x = 1
                    W_r[k*n+i, :] = x
            if not valid:
                continue
            
            # Verify
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                success = True
                best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                best_rank = r
                break
        
        if success:
            break
    
    return best_result, best_rank

def _try_structured(n, m, p, time_limit=10.0):
    """Try structured decompositions based on known constructions."""
    start = time.time()
    
    # Let's try a systematic approach for 2x2x6
    # n=2, m=2, p=6
    
    # The tensor T has T[i,j,k] = 1 for all i in [0,2), j in [0,2), k in [0,6)
    # We need: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Idea: use the 2x2x2 Strassen-like decomposition as a building block.
    # For 2x2x2, the minimum rank is 7 (Strassen).
    # For 2x2x6, we can think of it as three 2x2x2 blocks (for k=0,1; k=2,3; k=4,5)
    # But the W tensor couples different k values through the i index.
    
    # Let's try a direct construction.
    # Suppose we use r terms. For each term c, define:
    #   u_c: a vector in R^4 (indexed by (i,j))
    #   v_c: a vector in R^12 (indexed by (j,k))  
    #   w_c: a vector in R^12 (indexed by (k,i))
    
    # Constraint: for all (i,j,k) in [0,2)x[0,2)x[0,6):
    #   sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
    
    # Let's try a specific structure:
    # For each k, the "slice" T[:,:,k] is a 2x2 matrix of 1s.
    # So for fixed k: sum_c (sum_j u_c[i,j]*v_c[j,k]) * w_c[k,i] = 1 for all i.
    
    # Let a_c[i,k] = sum_j u_c[i,j] * v_c[j,k]  (this is like (U_c * V_c)[i,k])
    # Then: sum_c a_c[i,k] * w_c[k,i] = 1 for all (i,k)
    
    # This is a bilinear system. Let's try to construct it.
    
    # Strategy: choose u_c and v_c such that the a_c matrices are simple,
    # then solve for w_c.
    
    # Try rank 19: Use the "block" approach
    # Split k into 3 groups of 2: {0,1}, {2,3}, {4,5}
    # For each group, use a 2x2x2 decomposition of rank 7? No, that's for full 2x2x2.
    
    # Actually, let me just try the optimization approach with more attempts.
    
    best = None
    best_r = None
    
    for r in range(14, 25):
        if time.time() - start > time_limit:
            break
        
        found = False
        for seed in range(200):
            if time.time() - start > time_limit:
                break
            
            rng = np.random.RandomState(seed * 1000 + r)
            
            # Random U, V
            U_r = rng.randn(4, r) * 0.5
            V_r = rng.randn(12, r) * 0.5
            
            # For each (i,k), compute M[i,k,:] and solve for W
            M = np.zeros((2, 6, r))
            for i in range(2):
                for k in range(6):
                    for j in range(2):
                        M[i, k, :] += U_r[i*2+j, :] * V_r[j*6+k, :]
            
            W_r = np.zeros((12, r))
            valid = True
            for i in range(2):
                for k in range(6):
                    row = M[i, k, :]
                    norm_sq = row @ row
                    if norm_sq < 1e-12:
                        valid = False
                        break
                    W_r[k*2+i, :] = row / norm_sq
            if not valid:
                continue
            
            # Verify
            ok = True
            for i in range(2):
                for j in range(2):
                    for k in range(6):
                        val = 0
                        for c in range(r):
                            val += U_r[i*2+j, c] * V_r[j*6+k, c] * W_r[k*2+i, c]
                        if abs(val - 1) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                found = True
                best = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                best_r = r
                break
        
        if found:
            break
    
    return best, best_r

def _schoolbook_decomposition(n, m, p):
    """Fallback: schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank exact tensor decomposition for 2x2 by 2x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    total_start = time.time()
    deadline = 120.0  # Must finish within 120s
    
    # Fallback: schoolbook
    U_fb, V_fb, W_fb = _schoolbook_decomposition(n, m, p)
    
    # Try to find a better decomposition
    # First try structured/optimization approach
    try:
        result, rank = _try_structured(n, m, p, time_limit=110.0)
        if result is not None and rank < n*m*p:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Check time
    if time.time() - total_start > deadline - 5:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try the more general approach
    try:
        result, rank = _try_rank_decomposition(n, m, p, rank=14, time_limit=min(110.0, deadline - (time.time() - total_start) - 5))
        if result is not None and rank < n*m*p:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Return fallback
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
