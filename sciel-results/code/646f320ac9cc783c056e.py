import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Uses Strassen's algorithm on 6 independent 2x2 column-pairs.
    Total rank = 6 * 7 = 42.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    t_start = time.time()
    deadline = t_start + 115  # safety margin before 120s limit
    
    num_pairs = p // 2  # 6 pairs
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=float)   # 4 x 42
    V = np.zeros((m * p, rank), dtype=float)   # 24 x 42
    W = np.zeros((p * n, rank), dtype=float)   # 24 x 42
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                U[0, c] = 1
                U[3, c] = 1
                V[2*t, c] = 1
                V[p + 2*t+1, c] = 1
                W[2*t*n + 0, c] = 1
                W[(2*t+1)*n + 1, c] = 1
            elif s == 1:
                U[2, c] = 1
                U[3, c] = 1
                V[2*t, c] = 1
                W[2*t*n + 1, c] = 1
                W[(2*t+1)*n + 1, c] = -1
            elif s == 2:
                U[0, c] = 1
                V[2*t+1, c] = 1
                V[p + 2*t+1, c] = -1
                W[(2*t+1)*n + 0, c] = 1
                W[(2*t+1)*n + 1, c] = 1
            elif s == 3:
                U[3, c] = 1
                V[p + 2*t, c] = 1
                V[2*t, c] = -1
                W[2*t*n + 0, c] = 1
                W[2*t*n + 1, c] = 1
            elif s == 4:
                U[0, c] = 1
                U[1, c] = 1
                V[p + 2*t+1, c] = 1
                W[2*t*n + 0, c] = -1
                W[(2*t+1)*n + 0, c] = 1
            elif s == 5:
                U[2, c] = 1
                U[0, c] = -1
                V[2*t, c] = 1
                V[2*t+1, c] = 1
                W[(2*t+1)*n + 1, c] = 1
            elif s == 6:
                U[1, c] = 1
                U[3, c] = -1
                V[p + 2*t, c] = 1
                V[p + 2*t+1, c] = 1
                W[2*t*n + 0, c] = 1
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            # Fall back to schoolbook
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school), dtype=float)
            V_s = np.zeros((m * p, rank_school), dtype=float)
            W_s = np.zeros((p * n, rank_school), dtype=float)
            c = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_s[ii*m+jj, c] = 1
                        V_s[jj*p+kk, c] = 1
                        W_s[kk*n+ii, c] = 1
                        c += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
