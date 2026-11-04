import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 53: 49 for 4x4x4 Strassen, 4 for 4x4x1 schoolbook
    R = 53
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Recursive Strassen (Cols 0-48) ---
    
    # Local 2x2x2 Strassen matrices
    # U_S: 4 rows (i',j'), 7 cols (t)
    # V_S: 4 rows (j',k'), 7 cols (t)
    # W_S: 4 rows (k',i'), 7 cols (t)
    # Order of rows in W_S: (k'=0,i'=0), (k'=0,i'=1), (k'=1,i'=0), (k'=1,i'=1)
    # Indices: 0, 1, 2, 3
    
    U_S = [
        [1, 0, 1, 0, 1, -1, 0], # (0,0)
        [0, 0, 0, 0, 1, 0, 1],  # (0,1)
        [0, 1, 0, 0, 0, 1, 0],  # (1,0)
        [1, 1, 0, 1, 0, 0, -1]  # (1,1)
    ]
    
    V_S = [
        [1, 1, 0, -1, 0, 1, 0], # (0,0)
        [0, 0, 1, 0, 0, 1, 0],  # (0,1)
        [0, 0, 0, 1, 0, 0, 1],  # (1,0)
        [1, 0, -1, 0, 1, 0, 1]  # (1,1)
    ]
    
    W_S = [
        [1, 0, 0, 1, -1, 0, 1], # (0,0) -> C00
        [0, 0, 1, 0, 1, 0, 0],  # (0,1) -> C01
        [0, 1, 0, 1, 0, 0, 0],  # (1,0) -> C10
        [1, -1, 1, 0, 0, 1, 1]  # (1,1) -> C11
    ]
    
    # Coefficients for the 7 Strassen steps (s=0..6)
    # alpha[s][p][q] for A_pq
    # beta[s][q][r] for B_qr
    # delta[s][p][r] for C_pr contribution
    
    alpha = [
        [[1, 0], [0, 1]], # s=0: A00 + A11
        [[0, 0], [1, 1]], # s=1: A10 + A11
        [[1, 0], [0, 0]], # s=2: A00
        [[0, 0], [0, 1]], # s=3: A11
        [[1, 1], [0, 0]], # s=4: A00 + A01
        [[-1, 0], [1, 0]],# s=5: A10 - A00
        [[0, 1], [0, -1]] # s=6: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]], # s=0: B00 + B11
        [[1, 0], [0, 0]], # s=1: B00
        [[0, 1], [0, -1]],# s=2: B01 - B11
        [[-1, 0], [1, 0]],# s=3: B10 - B00
        [[0, 0], [0, 1]], # s=4: B11
        [[1, 1], [0, 0]], # s=5: B00 + B01
        [[0, 0], [1, 1]]  # s=6: B10 + B11
    ]
    
    delta = [
        [[1, 0], [0, 1]], # s=0: C00 + C11
        [[0, 0], [1, -1]],# s=1: C10 - C11
        [[0, 1], [0, 1]], # s=2: C01 + C11
        [[1, 0], [1, 0]], # s=3: C00 + C10
        [[-1, 1], [0, 0]],# s=4: -C00 + C01
        [[0, 0], [0, 1]], # s=5: C11
        [[1, 0], [0, 1]]  # s=6: C00 + C11
    ]
    
    # Fill U, V, W for the 49 terms
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            
            # Iterate over all i, j, k in 4x4x4
            for i in range(4):
                for j in range(4):
                    p_i = i // 2
                    q_j = j // 2
                    i_p = i % 2
                    j_p = j % 2
                    
                    u_val = alpha[s][p_i][q_j] * U_S[i_p * 2 + j_p][t]
                    if u_val != 0:
                        U[i * 4 + j][c] = u_val
                        
            for j in range(4):
                for k in range(4):
                    q_j = j // 2
                    r_k = k // 2
                    j_p = j % 2
                    k_p = k % 2
                    
                    v_val = beta[s][q_j][r_k] * V_S[j_p * 2 + k_p][t]
                    if v_val != 0:
                        V[j * 5 + k][c] = v_val
                        
            for k in range(4):
                for i in range(4):
                    r_k = k // 2
                    p_i = i // 2
                    k_p = k % 2
                    i_p = i % 2
                    
                    w_val = delta[s][p_i][r_k] * W_S[k_p * 2 + i_p][t]
                    if w_val != 0:
                        W[k * 4 + i][c] = w_val

    # --- Part 2: 4x4x1 Schoolbook for 5th column (Cols 49-52) ---
    # B column index 4
    for i in range(4):
        for j in range(4):
            c = 49 + i * 4 + j
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            W[4 * 4 + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
