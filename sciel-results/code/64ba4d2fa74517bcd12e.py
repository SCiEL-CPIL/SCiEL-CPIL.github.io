import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = 4, 4, 6
    
    # Strassen coefficients for 2x2 x 2x2
    # alpha: U coefficients (4 A-blocks -> 7 terms)
    # beta: V coefficients (4 B-blocks -> 7 terms)  
    # recon: reconstruction (7 terms -> 4 C-blocks)
    
    alpha = [
        [1, 0, 0, 1],   # s0: A00+A11
        [0, 0, 1, 1],   # s1: A10+A11
        [1, 0, 0, 0],   # s2: A00
        [0, 0, 0, 1],   # s3: A11
        [1, 1, 0, 0],   # s4: A00+A01
        [-1, 0, 1, 0],  # s5: A10-A00
        [0, 1, 0, -1]   # s6: A01-A11
    ]
    beta = [
        [1, 0, 0, 1],   # s0: B00+B11
        [1, 0, 0, 0],   # s1: B00
        [0, 1, 0, -1],  # s2: B01-B11
        [-1, 0, 1, 0],  # s3: B10-B00
        [0, 0, 0, 1],   # s4: B11
        [1, 1, 0, 0],   # s5: B00+B01
        [0, 0, 1, 1]    # s6: B10+B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # C00
        [(1, 0, 1), (1, 1, -1)],    # C10
        [(0, 1, 1), (1, 1, 1)],     # C01
        [(0, 0, 1), (1, 0, 1)],     # C11
        [(0, 0, -1), (0, 1, 1)],    # C00
        [(1, 1, 1)],                # C11
        [(0, 0, 1)]                 # C00
    ]
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        if time.time() > deadline:
            break
        # Part 1: Inner Strassen for 2x2 x 2x2 (cols 0,1 of Y_s)
        for si in range(7):
            # U: combine top-level alpha[s] and inner alpha[si]
            for a in range(4):
                if alpha[si][a] == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    if alpha[s][r] == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V: combine top-level beta[s] and inner beta[si]
            for b in range(4):
                if beta[si][b] == 0:
                    continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    if beta[s][c] == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W: combine reconstructions
            for (a, b, gamma) in recon[s]:
                for (za, zb, zgamma) in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += alpha[s][r]
                
                for c in range(4):
                    if beta[s][c] == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += beta[s][c]
                
                for (a, b, gamma) in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
