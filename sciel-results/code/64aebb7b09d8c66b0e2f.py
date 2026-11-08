import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)  
    # W: (p*n, R) = (24, R)
    
    # Top-level Strassen: 7 multiplications of (2x2) x (2x3)
    # For each (2x2)x(2x3): use inner Strassen for 2x2x2 (7 mults) + schoolbook for 2x2x1 (4 mults) = 11
    # Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen for 2x2 x 2x2
    s2_alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    s2_beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s2_recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        # 7 Strassen multiplications for the 2x2 x 2x2 part (cols 0,1 of the 2x3 block)
        for si in range(7):
            if time.time() > deadline - 5:
                break
            
            # U: combine top-level alpha[s] with inner alpha[si]
            for a in range(4):
                if s2_alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if s2_alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += s2_alpha[si][a] * s2_alpha[s][r]
            
            # V: combine top-level beta[s] with inner beta[si]
            # Inner beta applies to Y_s[:, 0:2] which is a 2x2 matrix
            # b indexes position in 2x2: row=b//2, col=b%2
            for b in range(4):
                if s2_beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if s2_beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            # B_c is 2x3. We take columns 0 and 1.
                            # So v_idx uses y_inner_col (0 or 1)
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += s2_beta[si][b] * s2_beta[s][c]
            
            # W: combine reconstructions
            # Inner output is 2x2. za, zb are row, col in 2x2.
            # Top-level output block (a,b) is 2x3.
            # Inner output corresponds to cols 0,1 of the 2x3 block.
            for a, b, gamma in s2_recon[s]:
                for za, zb, zgamma in s2_recon[si]:
                    # w_idx = (col_in_C) * 4 + (row_in_C)
                    # col_in_C = b * 3 + zb (since zb is 0 or 1)
                    # row_in_C = a * 2 + za
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # 4 schoolbook multiplications for the 2x2 x 2x1 part (col 2 of the 2x3 block)
        for u in range(2):
            for v in range(2):
                if time.time() > deadline - 5:
                    break
                
                # U: select rows from the 2x2 blocks based on top-level alpha[s]
                # Term: X_s[u,v] * Y_s[v,2]
                # X_s[u,v] is a linear combination of A blocks
                for r in range(4):
                    if s2_alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += s2_alpha[s][r]
                
                # V: select from 3rd column of the 2x3 blocks
                # Y_s[v,2] is a linear combination of B blocks
                for c in range(4):
                    if s2_beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        # B_c[v, 2]
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += s2_beta[s][c]
                
                # W: output contribution
                # C_ab[u, 2]
                for a, b, gamma in s2_recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
