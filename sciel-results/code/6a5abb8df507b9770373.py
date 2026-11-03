"""
Rewritten program for Matrix Multiplication 2x3x8.
Target: Rank 44 decomposition using Strassen for the 2x2x8 part and schoolbook for the 2x1x8 part.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly represents matrix multiplication."""
    if not (isinstance(U, np.ndarray) and isinstance(V, np.ndarray) and isinstance(W, np.ndarray)):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
    else:
        U = U.astype(float)
        V = V.astype(float)
        W = W.astype(float)
        
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    
    # Efficient vectorized check
    # X[r] = sum_j U[i*m+j, r] A[i,j] for all i? No, X depends on i.
    # Let's just do the direct sum for reliability, it's fast enough for small matrices.
    for r in range(rank):
        # Compute vector x of length n: x_i = sum_j U[i*m+j, r] * A[i,j]
        x = np.zeros(n)
        for i in range(n):
            for j in range(m):
                x[i] += U[i * m + j, r] * A[i, j]
        
        # Compute vector y of length p: y_k = sum_j V[j*p+k, r] * B[j,k]
        y = np.zeros(p)
        for k in range(p):
            for j in range(m):
                y[k] += V[j * p + k, r] * B[j, k]
        
        # Add contribution: C += W[:, r] outer x * y
        # W index is k*n + i
        for k in range(p):
            for i in range(n):
                C_computed[i, k] += W[k * n + i, r] * x[i] * y[k]
                
    return np.allclose(C_computed, C_expected, atol=tol)

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    # C01 += M3, C11 += M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    # C00 += M4, C10 += M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank_44():
    """
    Construct the rank 44 decomposition.
    n=2, m=3, p=8.
    Split m into m1=2 and m2=1.
    Part 1: A (2x2) * B (2x8). Use Strassen on 4 blocks of B (2x2 each).
    Part 2: A (2x1) * B (1x8). Schoolbook.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n * m, rank)) # 6 x 44
    V = np.zeros((m * p, rank)) # 24 x 44
    W = np.zeros((p * n, rank)) # 16 x 44
    
    S_U, S_V, S_W = _strassen_2x2()
    
    # Part 1: 2x2x8 using Strassen
    # A indices: i in 0..1, j in 0..1. U index: i*3 + j
    # B indices: j in 0..1, k in 0..7. V index: j*8 + k
    # C indices: i in 0..1, k in 0..7. W index: k*2 + i
    
    num_blocks = 4
    block_size = 2 # p is 8, so 4 blocks of 2
    
    for b in range(num_blocks):
        base_k = b * block_size
        for s in range(7):
            c = b * 7 + s # column index in U, V, W
            
            # U: A is 2x2.
            for i in range(2):
                for j in range(2):
                    # S_U index is i*2 + j
                    U[i * 3 + j, c] = S_U[i * 2 + j, s]
                    
            # V: B block is 2x2. Rows j in 0..1. Cols k in base_k..base_k+1.
            for j in range(2):
                for dk in range(2):
                    k = base_k + dk
                    # S_V index is j*2 + dk
                    V[j * 8 + k, c] = S_V[j * 2 + dk, s]
                    
            # W: C block is 2x2. Rows k in base_k..base_k+1. Cols i in 0..1.
            for dk in range(2):
                k = base_k + dk
                for i in range(2):
                    # S_W index is dk*2 + i (where dk corresponds to local k index)
                    W[k * 2 + i, c] = S_W[dk * 2 + i, s]
                    
    # Part 2: 2x1x8 using schoolbook
    # A indices: i in 0..1, j=2. U index: i*3 + 2
    # B indices: j=2, k in 0..7. V index: 2*8 + k
    # C indices: i in 0..1, k in 0..7. W index: k*2 + i
    
    base_col = 28
    for i in range(2):
        for k in range(8):
            c = base_col + i * 8 + k
            U[i * 3 + 2, c] = 1
            V[2 * 8 + k, c] = 1
            W[k * 2 + i, c] = 1
            
    return U, V, W

def _construct_schoolbook():
    """
    Construct the rank 48 schoolbook decomposition.
    """
    n, m, p = 2, 3, 8
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    # Try rank 44 first
    try:
        U, V, W = _construct_rank_44()
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass # Fall through to schoolbook
    
    # Fallback to rank 48
    U, V, W = _construct_schoolbook()
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # If even schoolbook fails (should not happen), return empty or raise
    # But we must return a dict.
    return {"U": [], "V": [], "W": []}

# EVOLVE-BLOCK-END
