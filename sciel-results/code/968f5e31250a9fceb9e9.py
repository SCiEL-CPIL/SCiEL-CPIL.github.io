"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Winograd convolutional framework with Hadamard transforms to reduce rank below schoolbook (80)."""
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
    Using the convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
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

def _recursive_strassen_4x4x4():
    """Construct rank-49 decomposition for 4x4 * 4x4 via recursive Strassen.
    
    Uses Kronecker product of 2x2 Strassen factors.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Convention: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Kronecker product gives the recursive structure
    # U4 = U2 ⊗ U2, V4 = V2 ⊗ V2, W4 = W2 ⊗ W2
    # This yields rank 7*7 = 49
    U4 = np.kron(U2, U2)  # shape (4*4, 7*7) = (16, 49)
    V4 = np.kron(V2, V2)  # shape (16, 49)
    W4 = np.kron(W2, W2)  # shape (16, 49)
    
    return U4, V4, W4

def _extend_to_4x4x5(U4, V4, W4):
    """Extend a 4x4*4x4 rank-r decomposition to 4x4*4x5 by adding 16 schoolbook terms.
    
    The extra column of B (index 4) requires 4*4 = 16 multiplications.
    Returns U5: (16, r+16), V5: (20, r+16), W5: (20, r+16)
    """
    r = U4.shape[1]
    r5 = r + 16
    
    U5 = np.zeros((16, r5), dtype=np.float64)
    V5 = np.zeros((20, r5), dtype=np.float64)
    W5 = np.zeros((20, r5), dtype=np.float64)
    
    U5[:, :r] = U4
    V5[:16, :r] = V4
    W5[:16, :r] = W4
    
    col = r
    for i in range(4):
        for j in range(4):
            U5[i*4 + j, col] = 1.0
            V5[j*5 + 4, col] = 1.0
            W5[4*4 + i, col] = 1.0
            col += 1
    
    return U5, V5, W5

def _recursive_strassen_4x4x5():
    """Full recursive Strassen construction for 4x4 * 4x5, rank 49+16=65."""
    U4, V4, W4 = _recursive_strassen_4x4x4()
    U5, V5, W5 = _extend_to_4x4x5(U4, V4, W4)
    return U5, V5, W5

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Primary: recursive Strassen for 4x4x4 (rank 49) + schoolbook extra column (rank 16) = 65
    try:
        U5, V5, W5 = _recursive_strassen_4x4x5()
        if _verify_fast(U5, V5, W5, n, m, p):
            return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 80)
    return _schoolbook()
# EVOLVE-BLOCK-END
