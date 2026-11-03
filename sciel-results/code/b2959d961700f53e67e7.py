import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 43 decomposition
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_00, w_01, w_10, w_11, c):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[u_a][c] += 1
        U[u_d][c] += 1
        V[v_e][c] += 1
        V[v_h][c] += 1
        W[w_00][c] += 1
        W[w_11][c] += 1
        
        # M2 = (c+d)e
        U[u_c][c] += 1
        U[u_d][c] += 1
        V[v_e][c] += 1
        W[w_10][c] += 1
        
        # M3 = a(f-h)
        U[u_a][c] += 1
        V[v_f][c] += 1
        V[v_h][c] += -1
        W[w_01][c] += 1
        W[w_11][c] += -1
        
        # M4 = d(g-e)
        U[u_d][c] += 1
        V[v_g][c] += 1
        V[v_e][c] += -1
        W[w_00][c] += -1
        W[w_10][c] += 1
        
        # M5 = (a+b)h
        U[u_a][c] += 1
        U[u_b][c] += 1
        V[v_h][c] += 1
        W[w_00][c] += 1
        W[w_01][c] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][c] += 1
        U[u_a][c] += -1
        V[v_e][c] += 1
        V[v_f][c] += 1
        W[w_11][c] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][c] += 1
        U[u_d][c] += -1
        V[v_g][c] += 1
        V[v_h][c] += 1
        W[w_00][c] += -1
        W[w_11][c] += -1
        
        col += 1

    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # A1 rows in U: 0, 1 (row 0), 5, 6 (row 1)
    # B1 rows in V: 0..5 (row 0), 6..11 (row 1)
    # Split B1 into 3 2x2 blocks (k=0,1; 2,3; 4,5)
    
    # B1,1 (k=0,1)
    add_strassen(0, 1, 5, 6, 
                 0, 1, 6, 7, 
                 0, 2, 1, 3, col)
    
    # B1,2 (k=2,3)
    add_strassen(0, 1, 5, 6, 
                 2, 3, 8, 9, 
                 4, 6, 5, 7, col)
                 
    # B1,3 (k=4,5)
    add_strassen(0, 1, 5, 6, 
                 4, 5, 10, 11, 
                 8, 10, 9, 11, col)

    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    # A2 rows in U: 2, 3 (row 0), 7, 8 (row 1)
    # B2 rows in V: 12..17 (row 2), 18..23 (row 3)
    
    # B2,1 (k=0,1)
    add_strassen(2, 3, 7, 8, 
                 12, 13, 18, 19, 
                 0, 2, 1, 3, col)
                 
    # B2,2 (k=2,3)
    add_strassen(2, 3, 7, 8, 
                 14, 15, 20, 21, 
                 4, 6, 5, 7, col)
                 
    # B2,3 (k=4,5)
    add_strassen(2, 3, 7, 8, 
                 16, 17, 22, 23, 
                 8, 10, 9, 11, col)

    # Block 3: A3 (col 4) * B3 (row 4)
    # A3 rows in U: 4 (row 0), 9 (row 1)
    # B3 rows in V: 24..29 (row 4)
    # This is rank 1
    U[4][col] += 1
    U[9][col] += 1
    for k in range(6):
        V[24 + k][col] += 1
        W[k * 2 + 0][col] += 1
        W[k * 2 + 1][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
