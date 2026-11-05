# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin before 120s limit

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    
    # Precompute indices for speed
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i * 4 + j, c] = U2[(i // 2) * 2 + (j // 2), c0] * U2[(i % 2) * 2 + (j % 2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j * 4 + k, c] = V2[(j // 2) * 2 + (k // 2), c0] * V2[(j % 2) * 2 + (k % 2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k * 4 + i, c] = W2[(k // 2) * 2 + (i // 2), c0] * W2[(k % 2) * 2 + (i % 2), c1]
    return U4, V4, W4

def _build_4x4x8_strassen():
    """Build 4x4x8 via two independent 4x4x4 Strassen-Kronecker blocks. Rank 98."""
    U4, V4, W4 = _build_4x4x4_tensors()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: Columns 0-3 of output
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]
            
    # Block 1: Columns 4-7 of output
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]
            
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8, seed=42, tol=1e-9):
    """Verify the tensor decomposition on random test matrices."""
    np.random.seed(seed)
    A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
    B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    rank = U.shape[1]
    C_comp = np.zeros((n, p), dtype=np.float64)
    for r in range(rank):
        Ar = A * U[:, r].reshape(n, m)
        Br = B * V[:, r].reshape(m, p)
        Mr = Ar @ Br
        Wr = W[:, r].reshape(p, n).T
        C_comp += Mr * Wr
    err = np.max(np.abs(C_comp - C_expected))
    return err < tol, err

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    U, V, W = _build_4x4x8_strassen()
    
    # Verify
    ok, err = _verify(U, V, W)
    if not ok:
        # Fallback to schoolbook if Strassen fails (should not happen)
        n, m, p = 4, 4, 8
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=np.int16)
        V_f = np.zeros((m * p, rank), dtype=np.int16)
        W_f = np.zeros((p * n, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
