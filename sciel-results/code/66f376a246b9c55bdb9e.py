"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Returns a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the border terms.
    """
    n, m, p = DIMENSIONS
    R = 26
    
    # Initialize matrices
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Helper to get flattened indices
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
        
    # --- Strassen Part (7 terms) for indices in {0, 1} ---
    # m1 = (a00 + a11)(b00 + b11)
    U[idx_u(0, 0), 0] = 1; U[idx_u(1, 1), 0] = 1
    V[idx_v(0, 0), 0] = 1; V[idx_v(1, 1), 0] = 1
    W[idx_w(0, 0), 0] = 1; W[idx_w(1, 1), 0] = 1
    
    # m2 = (a10 + a11)b00
    U[idx_u(1, 0), 1] = 1; U[idx_u(1, 1), 1] = 1
    V[idx_v(0, 0), 1] = 1
    W[idx_w(0, 1), 1] = 1; W[idx_w(1, 1), 1] = -1
    
    # m3 = a00(b01 - b11)
    U[idx_u(0, 0), 2] = 1
    V[idx_v(0, 1), 2] = 1; V[idx_v(1, 1), 2] = -1
    W[idx_w(1, 0), 2] = 1; W[idx_w(1, 1), 2] = 1
    
    # m4 = a11(b10 - b00)
    U[idx_u(1, 1), 3] = 1
    V[idx_v(1, 0), 3] = 1; V[idx_v(0, 0), 3] = -1
    W[idx_w(0, 0), 3] = 1; W[idx_w(0, 1), 3] = 1
    
    # m5 = (a00 + a01)b11
    U[idx_u(0, 0), 4] = 1; U[idx_u(0, 1), 4] = 1
    V[idx_v(1, 1), 4] = 1
    W[idx_w(0, 0), 4] = -1; W[idx_w(1, 0), 4] = 1
    
    # m6 = (a10 - a00)(b00 + b01)
    U[idx_u(1, 0), 5] = 1; U[idx_u(0, 0), 5] = -1
    V[idx_v(0, 0), 5] = 1; V[idx_v(0, 1), 5] = 1
    W[idx_w(1, 1), 5] = 1
    
    # m7 = (a01 - a11)(b10 + b11)
    U[idx_u(0, 1), 6] = 1; U[idx_u(1, 1), 6] = -1
    V[idx_v(1, 0), 6] = 1; V[idx_v(1, 1), 6] = 1
    W[idx_w(0, 0), 6] = 1
    
    # --- Standard Part (19 terms) for indices involving 2 ---
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
                
    # Verification
    U_r = U.reshape((n, m, R))
    V_r = V.reshape((m, p, R))
    W_r = W.reshape((p, n, R))
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p))
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook (Rank 27) if verification fails
        R = 27
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
