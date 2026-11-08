import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We aim for Rank 42: Kronecker product of M_2,2,2 (Rank 7) and M_2,2,3 (Rank 6)
    # If Rank 6 coefficients are not found, fallback to Rank 77 (Rank 7 x Rank 11)
    
    # Strassen coefficients for 2x2x2
    alpha_strassen = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta_strassen = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon_strassen = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    # Coefficients for M_2,2,3 Rank 6
    # A is 2x2, B is 2x3.
    # Indices: A[0,0], A[0,1], A[1,0], A[1,1]
    # B[0,0], B[0,1], B[0,2], B[1,0], B[1,1], B[1,2]
    # C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
    
    # This part is tricky. I will use the Rank 11 decomposition (Strassen + Schoolbook) 
    # to ensure correctness, resulting in Rank 77.
    # To attempt Rank 42, I would need the Rank 6 coefficients.
    # Given the risk, I will stick to the known correct Rank 77 but optimize the construction.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        if time.time() > deadline:
            break
        # Inner Strassen for 2x2x2 (cols 0,1 of Y)
        for si in range(7):
            for a in range(4):
                if alpha_strassen[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha_strassen[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha_strassen[si][a] * alpha_strassen[s][r]
            
            for b in range(4):
                if beta_strassen[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta_strassen[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta_strassen[si][b] * beta_strassen[s][c]
            
            for a, b, gamma in recon_strassen[s]:
                for za, zb, zgamma in recon_strassen[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Schoolbook for 2x2x1 (col 2 of Y)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha_strassen[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha_strassen[s][r]
                
                for c in range(4):
                    if beta_strassen[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta_strassen[s][c]
                
                for a, b, gamma in recon_strassen[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
