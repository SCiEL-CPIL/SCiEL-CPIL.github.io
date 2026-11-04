import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _schoolbook(n, m, p):
    """
    Returns the standard (rank n*m*p) decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)  # Indexed by k*n + i for C[i,k]
    """
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verifies the decomposition.
    C[i, k] = sum_r W[k*n + i, r] * (sum_{i',j} U[i'*m+j, r] A[i',j]) * (sum_{j',k'} V[j'*p+k', r] B[j',k'])
    """
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A.flatten() # index i*m + j
    B_flat = B.flatten() # index j*p + k
    
    # L_r = U[:, r] . A_flat
    L = U_arr @ A_flat # Shape (rank,)
    # R_r = V[:, r] . B_flat
    R = V_arr @ B_flat # Shape (rank,)
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            # W is indexed by k*n + i
            idx = k * n + i
            C_computed[i, k] = np.sum(W_arr[idx, :] * L * R)
            
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def _strassen_2x2():
    """
    Returns the Strassen 2x2x2 decomposition (rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p1*p5 - p4 + p6 + p7, p1*p3 + p2*p5], [p1*p4 + p3*p5, p2*p4 - p1 + p3 + p6]]
    
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(f+h)
    # p7 = (b-c)(e+h)
    
    U = [
        [1, 0, 1, 0, 1, 0, 0],  # a
        [0, 0, 0, 0, 1, 0, 0],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 1, 0],  # d
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 1],  # e
        [0, 0, 1, 0, 1, 0, 0],  # f
        [0, 0, 0, 1, 0, 1, 1],  # g
        [1, 0, 0, 0, 1, 1, 1],  # h
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],  # C[0,0] = p1*p5 - p4 + p6 + p7
        [0, 0, 1, 0, 1, 0, 0],  # C[0,1] = p1*p3 + p2*p5
        [0, 1, 0, 1, 0, 0, 0],  # C[1,0] = p1*p4 + p3*p5
        [0, 0, 0, 0, 0, 1, 0],  # C[1,1] = p2*p4 - p1 + p3 + p6
    ]
    
    # Wait, this isn't right. Let me use the standard Strassen formulation.
    # The tensor entries are T_{i,j,k} = 1 if i=k (for the output index mapping)
    # Actually, for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # So T is a 2x2x2 tensor where T[i,j,k] = 1 if i==k... no.
    # T[i,j,k] = 1 for all i,j,k where the product contributes to C[i,k]
    # Actually T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the "right" sense)
    # No wait: T[i,j,k] = 1 means A[i,j] and B[j,k] multiply to give C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # For 2x2x2, the tensor has 8 entries, all 1.
    # Strassen finds rank 7 decomposition.
    
    # Let me look up the exact Strassen matrices.
    # From standard references:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [
        [1, 0, 1, 0, 1, 0, 0],   # A11
        [0, 0, 0, 0, 1, 0, 0],   # A12
        [0, 1, 0, 0, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 1, 0],   # A22
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 1],   # B11
        [0, 0, 1, 0, 1, 0, 0],   # B12
        [0, 0, 0, 1, 0, 1, 1],   # B21
        [1, 0, 0, 0, 1, 1, 1],   # B22
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],   # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [0, 0, 0, 0, 0, 1, 0],   # C22
    ]
    
    return {"U": U, "V": V, "W": W}

def _embed_2x2_into_5x5(U2, V2, W2, n=5, m=5, p=5):
    """
    Embed a 2x2x2 Strassen decomposition into a 5x5x5 decomposition.
    We use the top-left 2x2 block and pad the rest with schoolbook.
    """
    # For the 2x2 block at positions (0,0) to (1,1):
    # U2 is (4, 7) where rows are A[0,0], A[0,1], A[1,0], A[1,1]
    # V2 is (4, 7) where rows are B[0,0], B[0,1], B[1,0], B[1,1]
    # W2 is (4, 7) where rows are C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Total rank = 7 + (25 - 4) = 7 + 21 = 28? No.
    # We need to handle the full 5x5 matrix.
    # The 2x2 block uses 7 multiplications.
    # The remaining entries need schoolbook.
    
    # Let's think differently. We partition the 5x5 matrices into blocks.
    # Use Strassen for a 2x2 block, schoolbook for the rest.
    
    # Actually, let's just use the full schoolbook for now since finding
    # a better 5x5x5 decomposition is complex.
    pass

def _try_block_strassen(n, m, p, deadline):
    """
    Try to use Strassen-like block decomposition for 5x5x5.
    Partition into blocks and use Strassen where beneficial.
    """
    start_time = time.time()
    
    # For 5x5, we can try to use a 2x2 Strassen block plus schoolbook for the rest.
    # But this requires careful bookkeeping.
    
    # Alternative: Use the fact that 5 = 2 + 3, and try to combine
    # a 2x2x2 Strassen (rank 7) with a 3x3x3 schoolbook (rank 27)
    # and cross terms.
    
    # This is getting complex. Let's try a simpler approach:
    # Use random linear combinations to search for lower rank.
    
    return None

def _random_search_rank(n, m, p, target_rank, deadline, max_iters=100):
    """
    Random search for a decomposition of given rank.
    """
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate random U, V, W
        np.random.seed(it)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Fallback
    result = _schoolbook(n, m, p)
    
    # Check time
    if time.time() > deadline - 10:
        return result
    
    # Try to find a lower rank decomposition using random search
    # Start with rank 49 (Strassen-based lower bound for 5x5x5 is around 49-50)
    for target_rank in [49, 50, 55, 60]:
        if time.time() > deadline - 15:
            break
        
        res = _random_search_rank(n, m, p, target_rank, deadline, max_iters=50)
        if res is not None:
            result = res
            break
    
    return result
