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
    
    # Use a direct approach: build the tensor and find a decomposition
    # For now, use the Strassen-based approach but verify correctness
    
    # Top-level: split 4x4 x 4x6 into 7 subproblems of 2x2 x 2x3
    # Each 2x2 x 2x3 can be done with 11 multiplications (7 Strassen for 2x2x2 + 4 for 2x2x1)
    # Total: 7 * 11 = 77
    
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
    
    # For 2x2 x 2x3: use Strassen for first 2 columns, schoolbook for 3rd column
    # This gives 7 + 4 = 11 multiplications
    
    col = 0
    for s in range(7):
        # 7 Strassen multiplications for the 2x2 x 2x2 part
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
            for b in range(4):
                if s2_beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if s2_beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += s2_beta[si][b] * s2_beta[s][c]
            
            # W: combine reconstructions
            for a, b, gamma in s2_recon[s]:
                for za, zb, zgamma in s2_recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # 4 schoolbook multiplications for the 2x2 x 2x1 part (3rd column)
        for u in range(2):
            for v in range(2):
                if time.time() > deadline - 5:
                    break
                
                # U: select rows from the 2x2 blocks based on top-level alpha[s]
                for r in range(4):
                    if s2_alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += s2_alpha[s][r]
                
                # V: select from 3rd column of the 2x3 blocks
                for c in range(4):
                    if s2_beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += s2_beta[s][c]
                
                # W: output to 3rd column
                for a, b, gamma in s2_recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                r = round(x)
                if abs(x - r) < 1e-9:
                    new_row.append(int(r))
                else:
                    r2 = round(2*x) / 2
                    if abs(x - r2) < 1e-9:
                        new_row.append(r2)
                    else:
                        new_row.append(float(x))
            result.append(new_row)
        return result
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
