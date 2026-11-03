import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # 4x4x4 Strassen (rank 49) for first 4 columns of B
    # 4x4x2 schoolbook (rank 16) for last 2 columns of B
    R = 65
    U = np.zeros((16, R))  # A is 4x4, indexed as A[i,j] -> row i*4+j
    V = np.zeros((24, R))  # B is 4x6, indexed as B[i,j] -> row i*6+j
    W = np.zeros((24, R))  # C is 4x6, W[k*n+i, r] where k=col, i=row
    
    # Strassen 2x2x2 base coefficients
    alpha_2 = [
        [1, 0, 0, 1],  # A00 + A11
        [0, 0, 1, 1],  # A10 + A11
        [1, 0, 0, 0],  # A00
        [0, 0, 0, 1],  # A11
        [1, 1, 0, 0],  # A00 + A01
        [-1, 0, 1, 0], # A10 - A00
        [0, 1, 0, -1]  # A01 - A11
    ]
    beta_2 = [
        [1, 0, 0, 1],  # B00 + B11
        [1, 0, 0, 0],  # B00
        [0, 1, 0, -1], # B01 - B11
        [-1, 0, 1, 0], # B10 - B00
        [0, 0, 0, 1],  # B11
        [1, 1, 0, 0],  # B00 + B01
        [0, 0, 1, 1]   # B10 + B11
    ]
    recon_2 = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    # 4x4x4 Strassen part: multiply A[4x4] * B[4x4] -> C[4x4]
    # A indexed: A[row, col] where row,col in 0..3, flat index = row*4+col
    # B indexed: B[row, col] where row in 0..3, col in 0..3, flat index = row*6+col
    # C indexed: C[row, col] where row in 0..3, col in 0..3, W index = col*4+row
    
    for s in range(7):  # outer Strassen step
        for si in range(7):  # inner Strassen step (for 2x2x2)
            # U coefficients for A
            for a in range(4):  # a indexes 2x2 block position (row, col)
                if alpha_2[si][a] != 0:
                    a_r, a_c = a // 2, a % 2
                    for r in range(4):  # r indexes outer 2x2 block position
                        if alpha_2[s][r] != 0:
                            r_r, r_c = r // 2, r % 2
                            # Full row = r_r*2 + a_r, full col = r_c*2 + a_c
                            u_idx = (r_r * 2 + a_r) * 4 + (r_c * 2 + a_c)
                            U[u_idx, col] += alpha_2[si][a] * alpha_2[s][r]
            
            # V coefficients for B (first 4 columns)
            for b in range(4):  # b indexes 2x2 block position in B
                if beta_2[si][b] != 0:
                    b_r, b_c = b // 2, b % 2
                    for c in range(4):  # c indexes outer 2x2 block position
                        if beta_2[s][c] != 0:
                            c_r, c_c = c // 2, c % 2
                            # Full row = c_r*2 + b_r, full col = c_c*2 + b_c
                            v_idx = (c_r * 2 + b_r) * 6 + (c_c * 2 + b_c)
                            V[v_idx, col] += beta_2[si][b] * beta_2[s][c]
            
            # W coefficients for C (first 4 columns)
            # C[row, col], W index = col*4 + row
            for a_idx, b_idx, gamma in recon_2[s]:
                for za, zb, zgamma in recon_2[si]:
                    # Full row = a_idx*2 + za, full col = b_idx*2 + zb
                    w_row = a_idx * 2 + za
                    w_col = b_idx * 2 + zb
                    w_idx = w_col * 4 + w_row
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
    
    # 4x4x2 schoolbook part: A[4x4] * B[4x2] -> C[4x2]
    # B columns 4,5; C columns 4,5
    for i in range(4):  # row of A/C
        for j in range(4):  # col of A / row of B
            for k in range(2):  # col of B/C (0,1 -> actual cols 4,5)
                u_idx = i * 4 + j
                v_idx = j * 6 + (4 + k)
                w_idx = (4 + k) * 4 + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
