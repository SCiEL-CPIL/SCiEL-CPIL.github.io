import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _strassen_222():
    """Return Strassen 2x2x2 decomposition as (U, V, W) each of shape (4, 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A11-A00)(B00+B10)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x5x6_strassen():
    """Build rank-54 decomposition for 2x5x6 matrix multiplication."""
    n, m, p = 2, 5, 6
    rank = 54
    
    U = np.zeros((n * m, rank))  # (10, 54)
    V = np.zeros((m * p, rank))  # (30, 54)
    W = np.zeros((p * n, rank))  # (12, 54)
    
    col = 0
    StrU, StrV, StrW = _strassen_222()
    
    # Block 1: A[:, 0:2] * B[0:2, :] -> 2x2x6
    for k_offset in [0, 2, 4]:
        for t in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = StrU[i * 2 + j, t]
            
            for j in range(2):
                for k in range(2):
                    V[j * p + (k_offset + k), col] = StrV[j * 2 + k, t]
            
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k) * n + i, col] = StrW[k * 2 + i, t]
            
            col += 1
    
    # Block 2: A[:, 2:4] * B[2:4, :] -> 2x2x6
    for k_offset in [0, 2, 4]:
        for t in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + (2 + j), col] = StrU[i * 2 + j, t]
            
            for j in range(2):
                for k in range(2):
                    V[(2 + j) * p + (k_offset + k), col] = StrV[j * 2 + k, t]
            
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k) * n + i, col] = StrW[k * 2 + i, t]
            
            col += 1
    
    # Block 3: A[:, 4:5] * B[4:5, :] -> 2x1x6, schoolbook
    for i in range(2):
        for k in range(6):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=5, p=6, num_trials=5):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    for _ in range(num_trials):
        A = np.random.randint(0, 10, size=(n, m)).astype(float)
        B = np.random.randint(0, 10, size=(m, p)).astype(float)
        C_true = A @ B
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for t in range(U.shape[1]):
                    u_val = 0.0
                    for j in range(m):
                        u_val += U[i * m + j, t] * A[i, j]
                    v_val = 0.0
                    for j in range(m):
                        v_val += V[j * p + k, t] * B[j, k]
                    C_calc[i, k] += W[k * n + i, t] * u_val * v_val
        
        if not np.allclose(C_true, C_calc, atol=1e-8):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    try:
        U, V, W = _build_2x5x6_strassen()
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
