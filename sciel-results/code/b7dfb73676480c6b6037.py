"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Kronecker-product Strassen (rank 49) for 4x4x4 plus schoolbook (rank 4) for 4x4x1 strip.
Total rank 53. Falls back to schoolbook (rank 80) if verification fails."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_fast(U, V, W, n=4, m=4, p=5):
    """Fast verification using numpy with float64, checking tolerance."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=1e-9)

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 as factor matrices.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7), dtype=np.float64)
    U[0, 0] = 1; U[0, 2] = 1; U[0, 4] = 1; U[0, 5] = -1
    U[1, 2] = 1; U[1, 4] = 1; U[1, 6] = 1
    U[2, 1] = 1; U[2, 5] = 1
    U[3, 0] = 1; U[3, 1] = 1; U[3, 3] = 1; U[3, 5] = 1; U[3, 6] = -1
    
    V = np.zeros((4, 7), dtype=np.float64)
    V[0, 0] = 1; V[0, 1] = 1; V[0, 3] = -1; V[0, 5] = 1
    V[1, 0] = 1; V[1, 2] = -1; V[1, 5] = 1
    V[2, 0] = 1; V[2, 3] = 1; V[2, 6] = 1
    V[3, 0] = 1; V[3, 4] = 1; V[3, 6] = 1
    
    W = np.zeros((4, 7), dtype=np.float64)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """Construct rank-49 Strassen decomposition for 4x4 * 4x4 via Kronecker product.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)
    
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            U[i * 4 + j, col] = U2[i0 * 2 + j0, c2] * U2[i1 * 2 + j1, c1]
            for j1 in range(2):
                for j0 in range(2):
                    for k1 in range(2):
                        for k0 in range(2):
                            j = j1 * 2 + j0
                            k = k1 * 2 + k0
                            V[j * 4 + k, col] = V2[j0 * 2 + k0, c2] * V2[j1 * 2 + k1, c1]
            for k1 in range(2):
                for k0 in range(2):
                    for i1 in range(2):
                        for i0 in range(2):
                            k = k1 * 2 + k0
                            i = i1 * 2 + i0
                            W[k * 4 + i, col] = W2[k0 * 2 + i0, c2] * W2[k1 * 2 + i1, c1]
    
    return U, V, W

def _strip_4x4x1():
    """Construct rank-4 schoolbook decomposition for 4x4 * 4x1 (5th column strip).
    U: (16, 4), V: (20, 4), W: (20, 4)
    Note: V and W have 20 rows because they're for the full 4x5 problem.
    """
    U = np.zeros((16, 4), dtype=np.float64)
    V = np.zeros((20, 4), dtype=np.float64)
    W = np.zeros((20, 4), dtype=np.float64)
    
    for j in range(4):
        for i in range(4):
            U[i * 4 + j, j] = 1
        V[j * 5 + 4, j] = 1  # B[j, 4]
        W[4 * 4 + i, j] = 1  # W index: k*n + i = 4*4 + i
    
    return U, V, W

def _assemble_4x4x5_kron():
    """Assemble 4x4x5 from 4x4x4 (Strassen rank 49) + 4x4x1 (schoolbook rank 4).
    Total rank: 53.
    """
    U444, V444, W444 = _strassen_4x4x4()
    U441, V441, W441 = _strip_4x4x1()
    
    total_rank = 53
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((20, total_rank), dtype=np.float64)
    W = np.zeros((20, total_rank), dtype=np.float64)
    
    # 4x4x4 part: columns 0..48
    # U444 is (16, 49) with indices i*4+j
    # V444 is (16, 49) with indices j*4+k
    # W444 is (16, 49) with indices k*4+i
    U[:, :49] = U444
    # V444[j*4+k, c] -> V[j*5+k, c]
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :49] = V444[j * 4 + k, :49]
    # W444[k*4+i, c] -> W[k*4+i, c] (same indexing since p=4 for this part)
    W[:16, :49] = W444
    
    # 4x4x1 part: columns 49..52
    U[:, 49:] = U441
    V[:, 49:] = V441
    W[:, 49:] = W441
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Primary: Kronecker Strassen 4x4x4 + schoolbook 4x4x1 = rank 53
    try:
        U, V, W = _assemble_4x4x5_kron()
        if _verify_fast(U, V, W, 4, 4, 5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to Schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
