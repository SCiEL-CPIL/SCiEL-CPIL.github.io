import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2 decomposition
    # u_indices: list of 4 global indices for A block (row-major)
    # v_indices: list of 4 global indices for B block (row-major)
    # w_indices: list of 4 global indices for C block (row-major, but W is k*n+i)
    def add_strassen(u_idx, v_idx, w_idx):
        nonlocal col
        # Strassen factors
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We define the 7 multiplications as pairs of linear forms in A and B
        # Each multiplication k has a vector a_k in R^4 and b_k in R^4
        # such that term is sum(a_k[i] A[i]) * sum(b_k[j] B[j])
        
        a1 = [1, 0, 0, 1] # A11+A22
        b1 = [1, 0, 0, 1] # B11+B22
        a2 = [0, 0, 1, 1] # A21+A22
        b2 = [1, 0, 0, 0] # B11
        a3 = [1, 0, 0, 0] # A11
        b3 = [0, 1, 0, -1] # B12-B22
        a4 = [0, 0, 0, 1] # A22
        b4 = [0, 0, 1, -1] # B21-B11
        a5 = [1, 1, 0, 0] # A11+A12
        b5 = [0, 0, 0, 1] # B22
        a6 = [0, 0, 1, -1] # A21-A11
        b6 = [1, 1, 0, 0] # B11+B12
        a7 = [0, 1, 0, -1] # A12-A22
        b7 = [0, 0, 1, 1] # B21+B22
        
        # C coefficients
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        c_coeffs = [
            [1, 0, 0, 1, -1, 0, 1],   # C11
            [0, 0, 1, 0, 1, 0, 0],    # C12
            [0, 1, 0, 1, 0, 0, 0],    # C21
            [1, -1, 1, 0, 0, 1, 0]    # C22
        ]
        
        a_vecs = [a1, a2, a3, a4, a5, a6, a7]
        b_vecs = [b1, b2, b3, b4, b5, b6, b7]
        
        for k in range(7):
            # Fill U
            for i in range(4):
                if a_vecs[k][i] != 0:
                    U[u_idx[i]][col] += a_vecs[k][i]
            
            # Fill V
            for j in range(4):
                if b_vecs[k][j] != 0:
                    V[v_idx[j]][col] += b_vecs[k][j]
            
            # Fill W
            for c in range(4):
                if c_coeffs[c][k] != 0:
                    W[w_idx[c]][col] += c_coeffs[c][k]
            
            col += 1

    # Part 1: A[0:4] B[0:4, 0:4]
    # A_L: cols 0,1 -> indices 0,1, 5,6
    # A_R: cols 2,3 -> indices 2,3, 7,8
    u_L = [0, 1, 5, 6]
    u_R = [2, 3, 7, 8]
    
    # B_LL: rows 0,1 cols 0,1 -> 0,1, 5,6
    # B_LR: rows 0,1 cols 2,3 -> 2,3, 7,8
    # B_RL: rows 2,3 cols 0,1 -> 10,11, 15,16
    # B_RR: rows 2,3 cols 2,3 -> 12,13, 17,18
    v_LL = [0, 1, 5, 6]
    v_LR = [2, 3, 7, 8]
    v_RL = [10, 11, 15, 16]
    v_RR = [12, 13, 17, 18]
    
    # C_L: cols 0,1 -> W indices k*2+i
    # k=0: i=0->0, i=1->1
    # k=1: i=0->2, i=1->3
    w_L = [0, 1, 2, 3]
    # C_R: cols 2,3
    # k=0: i=0->4, i=1->5
    # k=1: i=0->6, i=1->7
    w_R = [4, 5, 6, 7]
    
    add_strassen(u_L, v_LL, w_L)
    add_strassen(u_R, v_RL, w_L)
    add_strassen(u_L, v_LR, w_R)
    add_strassen(u_R, v_RR, w_R)
    
    # Part 2: A[4] B[4, 0:4]
    # A col 4: indices 4, 9
    # B row 4 cols 0-3: indices 20, 21, 22, 23
    # C cols 0-3: indices 0-7
    for i in range(2):
        for j in range(4):
            U[4 + i][col] = 1
            V[20 + j][col] = 1
            W[j * 2 + i][col] = 1
            col += 1
            
    # Part 3: A[0:4] B[0:4, 4]
    # A cols 0-3: indices 0-3, 5-8
    # B rows 0-3 col 4: indices 4, 9, 14, 19
    # C cols 4: indices 8, 9
    for i in range(2):
        for j in range(4):
            U[j + i*5][col] = 1 # A[i, j]
            V[j*5 + 4][col] = 1 # B[j, 4]
            W[4*2 + i][col] = 1 # C[i, 4] -> index 8+i
            col += 1
            
    # Part 4: A[4] B[4, 4]
    # A col 4: 4, 9
    # B row 4 col 4: 24
    # C col 4: 8, 9
    for i in range(2):
        U[4 + i][col] = 1
        V[24][col] = 1
        W[8 + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
