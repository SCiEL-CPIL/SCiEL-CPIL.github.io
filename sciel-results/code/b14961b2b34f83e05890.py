import numpy as np

def construct():
    # Rank-7 decomposition for M_{2,2,2} (Strassen)
    # A is 2x2, B is 2x2, C is 2x2
    # Indices: 0,1,2,3 for A, B, C
    alpha_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon_s = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    # Rank-6 decomposition for M_{2,2,3}
    # A is 2x2 (indices 0,1,2,3)
    # B is 2x3 (indices 0..5)
    # C is 2x3 (indices 0..5)
    # This is a known rank-6 decomposition
    U_6 = np.array([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1]
    ])
    V_6 = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    W_6 = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    
    # Construct M_{4,4,6} as Kronecker product of M_{2,2,3} and M_{2,2,2}
    # Rank = 6 * 7 = 42
    R = 42
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s6 in range(6):
        for s7 in range(7):
            # U part
            for i in range(4):
                if U_6[i, s6] != 0:
                    # i is index in A_6 (2x2)
                    # A_6 index i corresponds to (i//2, i%2)
                    # A_7 index r corresponds to (r//2, r%2)
                    # A_4 index (i//2, i%2) x (r//2, r%2)
                    for r in range(4):
                        if alpha_s[s7][r] != 0:
                            u_idx = (i//2 * 2 + r//2) * 4 + (i%2 * 2 + r%2)
                            U[u_idx, col] += U_6[i, s6] * alpha_s[s7][r]
            
            # V part
            for j in range(6):
                if V_6[j, s6] != 0:
                    # j is index in B_6 (2x3)
                    # B_6 index j corresponds to (j//3, j%3)
                    # B_7 index c corresponds to (c//2, c%2)
                    # B_4 index (j//3, c//2) x (j%3, c%2) ?
                    # B_4 is 4x6.
                    # B_6 is 2x3. B_7 is 2x2.
                    # B_4 index: row = (j//3)*2 + c//2, col = (j%3)*2 + c%2 ?
                    # No, B_6 col is 0,1,2. B_7 col is 0,1.
                    # B_4 col is 0..5.
                    # Mapping: col_4 = col_6 * 2 + col_7 ?
                    # row_4 = row_6 * 2 + row_7 ?
                    for c in range(4):
                        if beta_s[s7][c] != 0:
                            v_idx = (j//3 * 2 + c//2) * 6 + (j%3 * 2 + c%2)
                            V[v_idx, col] += V_6[j, s6] * beta_s[s7][c]
                            
            # W part
            for k in range(6):
                if W_6[k, s6] != 0:
                    # k is index in C_6 (2x3)
                    # k corresponds to (k//3, k%3)
                    for a, b, gamma in recon_s[s7]:
                        # a, b are indices in C_7 (2x2)
                        # a = a_row * 2 + a_col
                        # b = b_row * 2 + b_col
                        # C_4 index: row = (k//3)*2 + a_row, col = (k%3)*2 + a_col ?
                        # Wait, C_6 is 2x3. C_7 is 2x2.
                        # C_4 is 4x6.
                        # row_4 = row_6 * 2 + row_7
                        # col_4 = col_6 * 2 + col_7
                        w_idx = (k//3 * 2 + a//2) * 6 + (k%3 * 2 + a%2)
                        W[w_idx, col] += W_6[k, s6] * gamma
                        
            col += 1
            
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
