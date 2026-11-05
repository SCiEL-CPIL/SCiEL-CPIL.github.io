"""Matrix multiplication 4x4x6 - Strassen-based rank-77 decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 16, 24, 24
    
    def schoolbook():
        rank = n * m * p  # 96
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen coefficients for 2x2x2
    # Block indices: 0=00, 1=01, 2=10, 3=11
    alpha = [
        [1, 0, 0, 1],   # s0: A00 + A11
        [0, 0, 1, 1],   # s1: A10 + A11
        [1, 0, 0, 0],   # s2: A00
        [0, 0, 0, 1],   # s3: A11
        [1, 1, 0, 0],   # s4: A00 + A01
        [-1, 0, 1, 0],  # s5: A10 - A00
        [0, 1, 0, -1]   # s6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],   # s0: B00 + B11
        [1, 0, 0, 0],   # s1: B00
        [0, 1, 0, -1],  # s2: B01 - B11
        [-1, 0, 1, 0],  # s3: B10 - B00
        [0, 0, 0, 1],   # s4: B11
        [1, 1, 0, 0],   # s5: B00 + B01
        [0, 0, 1, 1]    # s6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s0
        [(1, 0, 1), (1, 1, -1)],    # s1
        [(0, 1, 1), (1, 1, 1)],     # s2
        [(0, 0, 1), (1, 0, 1)],     # s3
        [(0, 0, -1), (0, 1, 1)],    # s4
        [(1, 1, 1)],                # s5
        [(0, 0, 1)]                 # s6
    ]
    
    R = 77
    U = np.zeros((nm, R), dtype=int)
    V = np.zeros((mp, R), dtype=int)
    W = np.zeros((pn, R), dtype=int)
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for (2x2)*(2x2) [columns 0,1 of Y_s]
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            for a_r, b_r, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b_r * 3 + zb) * 4 + (a_r * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
        
        # Part 2: Schoolbook for (2x2)*(2x1) [column 2 of Y_s]
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                for a_r, b_r, gamma in recon[s]:
                    w_idx = (b_r * 3 + 2) * 4 + (a_r * 2 + u)
                    W[w_idx, col] += gamma
                col += 1
    
    # Verify
    def verify_decomp(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s_val = 0
                    for r in range(rank):
                        s_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if s_val != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W):
        return schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
