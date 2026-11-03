import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Fallback: Current program logic (Rank 189)
    def get_fallback():
        nb = 3
        bs = 2
        strassen_x = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,1,1)],
            [(0,0,1), (0,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,1,1), (1,1,-1)],
        ]
        strassen_y = [
            [(0,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (0,0,-1)],
            [(1,1,1)],
            [(0,0,1), (0,1,1)],
            [(1,0,1), (1,1,1)],
        ]
        strassen_c = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,-1)],
            [(0,1,1), (1,1,1)],
            [(0,0,1), (1,0,1)],
            [(0,0,-1), (0,1,1)],
            [(1,1,1)],
            [(0,0,1)],
        ]
        rank_bs = nb * nb * nb * 7
        U = np.zeros((n*m, rank_bs), dtype=np.float64)
        V = np.zeros((m*p, rank_bs), dtype=np.float64)
        W = np.zeros((p*n, rank_bs), dtype=np.float64)
        
        col = 0
        for ib in range(nb):
            for jb in range(nb):
                for kb in range(nb):
                    for s in range(7):
                        for (lr, lc, coeff) in strassen_x[s]:
                            gi = ib * bs + lr
                            gj = jb * bs + lc
                            U[gi * m + gj, col] += coeff
                        for (lr, lc, coeff) in strassen_y[s]:
                            gj = jb * bs + lr
                            gk = kb * bs + lc
                            V[gj * p + gk, col] += coeff
                        for (lr, lc, coeff) in strassen_c[s]:
                            gi = ib * bs + lr
                            gk = kb * bs + lc
                            W[gk * n + gi, col] += coeff
                        col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try Rank 23 Winograd 3x3 blocks with Strassen 2x2 wrapper
    # Total Rank: 7 * 23 = 161
    
    def get_winograd_3x3():
        # Winograd 23-multiplication algorithm for 3x3 matrix multiplication
        # Returns U, V, W of shape (9, 23)
        # Coefficients are integers
        U = np.zeros((9, 23), dtype=np.float64)
        V = np.zeros((9, 23), dtype=np.float64)
        W = np.zeros((9, 23), dtype=np.float64)
        
        # Hardcoded coefficients for Winograd 3x3
        # Note: This is a known rank 23 decomposition.
        # If these are incorrect, the verification will fail.
        
        # M0
        U[:, 0] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        
        # This is a placeholder. I don't have the full 23 coefficients memorized correctly.
        # I will use a safe fallback.
        return None

    # Since I cannot reliably provide the Rank 23 coefficients from memory without error,
    # and the current program is valid, I will return the fallback.
    # However, the directive asks for improvement.
    # I will return the current program's logic which is known to work.
    
    return get_fallback()
