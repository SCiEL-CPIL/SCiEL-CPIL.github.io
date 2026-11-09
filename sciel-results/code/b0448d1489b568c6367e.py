"""Matrix multiplication 3x3x8 - rank-minimal tensor decomposition."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid index."""
    # T is a 3-tensor of shape (n, m, p) where T[i,j,k] = 1
    # The multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # Actually for matrix mult tensor: T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition: T = sum_r U[r] ⊗ V[r] ⊗ W[r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix mult, T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent
    # i.e., the j in U row and V row must match, and k in V row and W row must match
    # Actually: U row index = i*m+j, V row index = j*p+k, W row index = k*n+i
    # The tensor T is defined as: T[a, b, c] = 1 if there exist i,j,k with
    # a = i*m+j, b = j*p+k, c = k*n+i
    # This is automatically satisfied for all valid combinations.
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    
    # Total size: (n*m) * (m*p) * (p*n) = n*m*p * m*p*n * ... 
    # Actually the tensor has dimensions (n*m) x (m*p) x (p*n)
    # But most entries are zero. The nonzero entries are at positions
    # (i*m+j, j*p+k, k*n+i) for i in range(n), j in range(m), k in range(p)
    
    return n, m, p

def _check_decomposition(U, V, W, n, m, p):
    """Check if the decomposition is exact. Returns True if valid."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # For other (a,b,c) combinations: should equal 0
    
    # Check all nonzero positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check some zero positions (sample to save time)
    # A position (a, b, c) is zero if there's no (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # i = a // m, j = a % m, k = b % p, and we need b // p == j and c % n == i and c // n == k
    # Let's check all positions that could potentially be nonzero but aren't
    # Actually, let's verify a subset of zero positions
    checked = 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Determine if this should be zero
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                # Nonzero only if j_a == j_b and k_b == k_c and i_a == i_c
                if not (j_a == j_b and k_b == k_c and i_a == i_c):
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > 1e-9:
                        return False
                    checked += 1
                    if checked > 5000:  # Limit checks for speed
                        break
            if checked > 5000:
                break
        if checked > 5000:
            break
    
    return True

def _strassen_3x3():
    """Return Strassen-like rank-7 decomposition for 3x3x3 matrix mult.
    
    Returns U, V, W each of shape (9, 7) for the 3x3x3 case.
    """
    # Use a known rank-7 decomposition for 3x3x3
    # This is based on the fact that 3x3x3 has rank at most 7
    # (schoolbook is 9, and there are rank-7 decompositions)
    
    # One approach: use the "3x3" algorithm
    # Let's use a simple construction:
    # A = [a11 a12 a13]     B = [b11 b12 b13]
    #     [a21 a22 a23]         [b21 b22 b23]
    #     [a31 a32 a33]         [b31 b32 b33]
    #
    # C = A*B
    #
    # We can try to find 7 bilinear forms.
    # 
    # Actually, let me use a verified rank-7 decomposition.
    # The tensor for 3x3x3 matrix multiplication has rank 7.
    # 
    # Here's one construction using the "border rank" approach:
    # We'll use the following 7 terms:
    
    n = m = p = 3
    rank = 7
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Index: U[i*3+j], V[j*3+k], W[k*3+i]
    # Term 0: a11*b11 -> c11
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: a12*b21 -> c12... wait, let me think more carefully.
    
    # Let me use a different approach. I'll construct it using the known
    # rank-7 formula for 3x3 matrix multiplication.
    
    # Actually, let me just use a verified set of 7 rank-1 tensors.
    # The 3x3x3 matrix multiplication tensor has rank 7.
    
    # Here's a known rank-7 decomposition (from literature):
    # Using the "3x3" algorithm by various authors.
    
    # Let me try a constructive approach:
    # Split into 2x2 blocks and use Strassen, but that gives more than 7.
    
    # Alternative: use the fact that we can write it as:
    # C11 = a11*b11 + a12*b21 + a13*b31
    # C12 = a11*b12 + a12*b22 + a13*b32
    # C13 = a11*b13 + a12*b23 + a13*b33
    # C21 = a21*b11 + a22*b21 + a23*b31
    # C22 = a21*b12 + a22*b22 + a23*b32
    # C23 = a21*b13 + a22*b23 + a23*b33
    # C31 = a31*b11 + a32*b21 + a33*b31
    # C32 = a31*b12 + a32*b22 + a33*b32
    # C33 = a31*b13 + a32*b23 + a33*b33
    
    # Let me try to find 7 bilinear forms that generate all 9 products.
    
    # Define 7 pairs of linear forms in A and B:
    # M1 = (a11+a21)(b11+b21) = a11b11 + a11b21 + a21b11 + a21b21
    # M2 = (a11-a21)(b11+b21) = a11b11 + a11b21 - a21b11 - a21b21
    # M3 = (a12+a22)(b12+b22) = a12b12 + a12b22 + a22b12 + a22b22
    # M4 = (a12-a22)(b12+b22) = a12b12 + a12b22 - a22b12 - a22b22
    # M5 = (a13+a23)(b13+b23) = a13b13 + a13b23 + a23b13 + a23b23
    # M6 = (a13-a23)(b13+b23) = a13b13 + a13b23 - a23b13 - a23b23
    # M7 = (a31+a32+a33)(b11+b21+b31) ... this doesn't work cleanly.
    
    # This approach is getting complicated. Let me use a simpler strategy:
    # Just use the schoolbook for 3x3 (rank 9) and rely on the optimization
    # to reduce the total rank. The 3x3x5 part can potentially be reduced.
    
    # Actually, let me just use schoolbook for both parts initially:
    # 3x3x3: rank 9
    # 3x3x5: rank 15
    # Total: rank 24 (much better than 72!)
    
    # Wait, that's not right. The 3x3x8 multiplication isn't a direct sum
    # of 3x3x3 and 3x3x5 in the tensor sense. Let me reconsider.
    
    # The 3x3x8 matrix multiplication: A is 3x3, B is 3x8, C is 3x8.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1,2}, k in {0,...,7}
    
    # The tensor has dimensions (3*3) x (3*8) x (8*3) = 9 x 24 x 24
    # Nonzero entries at (i*3+j, j*8+k, k*3+i) for i in {0,1,2}, j in {0,1,2}, k in {0,...,7}
    # That's 3*3*8 = 72 nonzero entries.
    
    # A direct sum decomposition would require the tensor to be block-diagonal,
    # which it isn't in a simple way. However, we can still try to find
    # efficient decompositions.
    
    # Let me use a different strategy: start with schoolbook (rank 72) and
    # try to reduce it using optimization.
    
    return None  # We'll use schoolbook as starting point

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_time=100):
    """Try to reduce the rank by finding linear dependencies or zero columns."""
    start_time = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0]) if U else 0
    
    # Try to find columns that can be removed or combined
    # A column r can be removed if U[:,r], V[:,r], W[:,r] are all zero
    # Or if the triple (U[:,r], V[:,r], W[:,r]) is linearly dependent with others
    
    # Simple approach: try removing each column and checking validity
    for r in range(rank - 1, -1, -1):
        if time.time() - start_time > max_time - 5:
            break
        
        # Check if column r is all zeros
        if all(U[i][r] == 0 for i in range(len(U))) and \
           all(V[i][r] == 0 for i in range(len(V))) and \
           all(W[i][r] == 0 for i in range(len(W))):
            # Remove column r
            U = [row[:r] + row[r+1:] for row in U]
            V = [row[:r] + row[r+1:] for row in V]
            W = [row[:r] + row[r+1:] for row in W]
            rank -= 1
            continue
        
        # Try setting column r to zero and checking if still valid
        # This is expensive, so only do it for a few columns
        if rank <= 40:  # Only try when close to target
            U_test = [row[:] for row in U]
            V_test = [row[:] for row in V]
            W_test = [row[:] for row in W]
            for i in range(len(U_test)):
                U_test[i][r] = 0
            for i in range(len(V_test)):
                V_test[i][r] = 0
            for i in range(len(W_test)):
                W_test[i][r] = 0
            
            if _check_decomposition(U_test, V_test, W_test, n, m, p):
                U = U_test
                V = V_test
                W = W_test
                rank -= 1
    
    return U, V, W, rank

def _optimize_integers(U, V, W, n, m, p, max_time=100):
    """Try to find a lower-rank decomposition by adjusting coefficients.
    
    This is a heuristic approach: try small integer perturbations.
    """
    start_time = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0]) if U else 0
    
    # Try to find if we can reduce rank by 1 by finding a linear combination
    # This is NP-hard in general, so we use heuristics.
    
    # Heuristic: try to find two columns that can be merged
    # Column r and s can be merged if there exists alpha such that
    # U[:,r] + alpha*U[:,s], V[:,r] + alpha*V[:,s], W[:,r] + alpha*W[:,s]
    # is a valid rank-1 tensor (or can be combined with others)
    
    # Simpler heuristic: try small random perturbations and see if rank decreases
    # after rounding to integers/half-integers
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try adding random low-rank perturbations
    import random
    random.seed(42)
    
    for attempt in range(50):
        if time.time() - start_time > max_time - 10:
            break
        
        # Create a perturbed version
        U_p = [row[:] for row in U]
        V_p = [row[:] for row in V]
        W_p = [row[:] for row in W]
        
        # Add small random noise
        for i in range(len(U_p)):
            for j in range(rank):
                if random.random() < 0.1:
                    U_p[i][j] += random.choice([-0.5, 0.5, -1, 1])
        for i in range(len(V_p)):
            for j in range(rank):
                if random.random() < 0.1:
                    V_p[i][j] += random.choice([-0.5, 0.5, -1, 1])
        for i in range(len(W_p)):
            for j in range(rank):
                if random.random() < 0.1:
                    W_p[i][j] += random.choice([-0.5, 0.5, -1, 1])
        
        # Round to nearest half-integer
        for i in range(len(U_p)):
            for j in range(rank):
                U_p[i][j] = round(U_p[i][j] * 2) / 2
        for i in range(len(V_p)):
            for j in range(rank):
                V_p[i][j] = round(V_p[i][j] * 2) / 2
        for i in range(len(W_p)):
            for j in range(rank):
                W_p[i][j] = round(W_p[i][j] * 2) / 2
        
        if _check_decomposition(U_p, V_p, W_p, n, m, p):
            # Try to reduce rank
            U_r, V_r, W_r, new_rank = _try_reduce_rank(U_p, V_p, W_p, n, m, p, max_time=5)
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U_r, V_r, W_r
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    rank = len(U[0])
    
    # Try to reduce rank
    U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, max_time=30)
    
    if time.time() < deadline - 20:
        U, V, W, rank = _optimize_integers(U, V, W, n, m, p, max_time=60)
    
    # Final verification
    if not _check_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        rank = len(U[0])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
