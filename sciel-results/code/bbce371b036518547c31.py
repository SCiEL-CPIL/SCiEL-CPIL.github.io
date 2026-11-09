"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # Rank 50 decomposition
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 B_1 (2x1 by 1x9)
    # A_1 is col 0 of A (indices 0, 3 in U)
    # B_1 is row 0 of B (indices 0..8 in V)
    # C is 2x9
    for i in range(n):
        for k in range(p):
            # U index: i*m + 0
            U[i * m + 0][col] = 1
            # V index: 0*p + k
            V[0 * p + k][col] = 1
            # W index: k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_2 B_2 (2x2 by 2x9)
    # A_2 is cols 1,2 of A
    # U indices: 
    # a (0,1) -> 0*m + 1 = 1
    # b (0,2) -> 0*m + 2 = 2
    # c (1,1) -> 1*m + 1 = 4
    # d (1,2) -> 1*m + 2 = 5
    
    # B_2 is rows 1,2 of B
    # V indices:
    # Row 1: 1*p + k = 9 + k
    # Row 2: 2*p + k = 18 + k
    
    # Strassen for 2x2 by 2x2 blocks
    # 4 blocks of 2x2, 1 block of 2x1
    
    def add_strassen_block(b_start):
        nonlocal col
        # b_start is the column index in B_2 (0, 2, 4, 6)
        # V indices for e, f, g, h
        # e = B[1, b_start] -> V[9 + b_start]
        # f = B[1, b_start+1] -> V[9 + b_start + 1]
        # g = B[2, b_start] -> V[18 + b_start]
        # h = B[2, b_start+1] -> V[18 + b_start + 1]
        
        v_e = 9 + b_start
        v_f = 9 + b_start + 1
        v_g = 18 + b_start
        v_h = 18 + b_start + 1
        
        # U indices
        u_a = 1
        u_b = 2
        u_c = 4
        u_d = 5
        
        # W indices for C[0, b_start], C[0, b_start+1], C[1, b_start], C[1, b_start+1]
        # C[0, k] -> W[k*n + 0] = W[2k]
        # C[1, k] -> W[k*n + 1] = W[2k+1]
        
        w_c11 = 2 * b_start
        w_c12 = 2 * (b_start + 1)
        w_c21 = 2 * b_start + 1
        w_c22 = 2 * (b_start + 1) + 1
        
        # M1: (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_c11][col] += 1
        W[w_c22][col] += 1
        col += 1
        
        # M2: (c+d)e
        U[u_c][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        W[w_c21][col] += 1
        W[w_c22][col] -= 1
        col += 1
        
        # M3: a(f-h)
        U[u_a][col] += 1
        V[v_f][col] += 1
        V[v_g][col] -= 1
        W[w_c12][col] += 1
        W[w_c22][col] += 1
        col += 1
        
        # M4: d(g-e)
        U[u_d][col] += 1
        V[v_g][col] += 1
        V[v_e][col] -= 1
        W[w_c11][col] += 1
        W[w_c21][col] += 1
        col += 1
        
        # M5: (a+b)h
        U[u_a][col] += 1
        U[u_b][col] += 1
        V[v_h][col] += 1
        W[w_c11][col] -= 1
        W[w_c12][col] += 1
        col += 1
        
        # M6: (c-a)(e+f)
        U[u_c][col] += 1
        U[u_a][col] -= 1
        V[v_e][col] += 1
        V[v_f][col] += 1
        W[w_c22][col] += 1
        col += 1
        
        # M7: (b-d)(g+h)
        U[u_b][col] += 1
        U[u_d][col] -= 1
        V[v_g][col] += 1
        V[v_h][col] += 1
        W[w_c11][col] += 1
        col += 1
        
    # Add 4 Strassen blocks
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    add_strassen_block(6)
    
    # Part 2b: 2x1 block (col 8)
    # e = B[1, 8] -> V[17]
    # g = B[2, 8] -> V[26]
    
    v_e = 17
    v_g = 26
    
    u_a = 1
    u_b = 2
    u_c = 4
    u_d = 5
    
    # C[0, 8] -> W[16]
    # C[1, 8] -> W[17]
    
    w_c1 = 16
    w_c2 = 17
    
    # ae
    U[u_a][col] += 1
    V[v_e][col] += 1
    W[w_c1][col] += 1
    col += 1
    
    # bg
    U[u_b][col] += 1
    V[v_g][col] += 1
    W[w_c1][col] += 1
    col += 1
    
    # ce
    U[u_c][col] += 1
    V[v_e][col] += 1
    W[w_c2][col] += 1
    col += 1
    
    # dg
    U[u_d][col] += 1
    V[v_g][col] += 1
    W[w_c2][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
