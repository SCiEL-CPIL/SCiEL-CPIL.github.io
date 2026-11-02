"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 54
    # 12 for 2x1x6, 42 for 2x4x6 (2 * 21)
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x1x6
    # i in {0,1}, j in {0}, k in {0..5}
    col = 0
    for i in range(n):
        for k in range(p):
            # U[i*m + j, col]
            U[i * m + 0][col] = 1
            # V[j*p + k, col]
            V[0 * p + k][col] = 1
            # W[k*n + i, col]
            W[k * n + i][col] = 1
            col += 1
    # col is now 12

    # Part 2: 2x4x6
    # j in {1,2,3,4}
    # Split into two 2x2x6 blocks: j in {1,2} and j in {3,4}
    
    def add_strassen_block(j_offset, k_offset, col_start):
        # j_offset: global index of first j in this 2x2 block (1 or 3)
        # k_offset: global index of first k in this 2x2x2 block (0, 2, or 4)
        # col_start: start column index in U,V,W
        
        # Strassen coefficients for 2x2x2
        # P1 = (a+d)(e+h)
        # P2 = (c+d)h
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)e
        # P6 = (c-a)(h+f)
        # P7 = (b-d)(e+g)
        
        # U (A matrix), V (B matrix), W (C matrix coeff)
        # Local indices i,j,k in {0,1}
        
        # P1
        c = col_start
        U[0 * m + (j_offset + 0)][c] = 1 # a
        U[1 * m + (j_offset + 1)][c] = 1 # d
        V[(j_offset + 0) * p + (k_offset + 0)][c] = 1 # e
        V[(j_offset + 1) * p + (k_offset + 1)][c] = 1 # h
        W[(k_offset + 0) * n + 0][c] = 1
        W[(k_offset + 1) * n + 0][c] = 0
        W[(k_offset + 0) * n + 1][c] = 0
        W[(k_offset + 1) * n + 1][c] = 1
        c += 1
        
        # P2
        U[1 * m + (j_offset + 0)][c] = 1 # c
        U[1 * m + (j_offset + 1)][c] = 1 # d
        V[(j_offset + 1) * p + (k_offset + 1)][c] = 1 # h
        W[(k_offset + 0) * n + 0][c] = 0
        W[(k_offset + 1) * n + 0][c] = 0
        W[(k_offset + 0) * n + 1][c] = 1
        W[(k_offset + 1) * n + 1][c] = -1
        c += 1
        
        # P3
        U[0 * m + (j_offset + 0)][c] = 1 # a
        V[(j_offset + 0) * p + (k_offset + 1)][c] = 1 # f
        V[(j_offset + 1) * p + (k_offset + 1)][c] = -1 # -h
        W[(k_offset + 0) * n + 0][c] = 0
        W[(k_offset + 1) * n + 0][c] = 0
        W[(k_offset + 0) * n + 1][c] = 1
        W[(k_offset + 1) * n + 1][c] = 1
        c += 1
        
        # P4
        U[1 * m + (j_offset + 1)][c] = 1 # d
        V[(j_offset + 1) * p + (k_offset + 0)][c] = 1 # g
        V[(j_offset + 0) * p + (k_offset + 0)][c] = -1 # -e
        W[(k_offset + 0) * n + 0][c] = 1
        W[(k_offset + 1) * n + 0][c] = 0
        W[(k_offset + 0) * n + 1][c] = 1
        W[(k_offset + 1) * n + 1][c] = 0
        c += 1
        
        # P5
        U[0 * m + (j_offset + 0)][c] = 1 # a
        U[0 * m + (j_offset + 1)][c] = 1 # b
        V[(j_offset + 0) * p + (k_offset + 0)][c] = 1 # e
        W[(k_offset + 0) * n + 0][c] = -1
        W[(k_offset + 1) * n + 0][c] = 0
        W[(k_offset + 0) * n + 1][c] = 1
        W[(k_offset + 1) * n + 1][c] = 0
        c += 1
        
        # P6
        U[1 * m + (j_offset + 0)][c] = 1 # c
        U[0 * m + (j_offset + 0)][c] = -1 # -a
        V[(j_offset + 1) * p + (k_offset + 1)][c] = 1 # h
        V[(j_offset + 0) * p + (k_offset + 1)][c] = 1 # f
        W[(k_offset + 0) * n + 0][c] = 0
        W[(k_offset + 1) * n + 0][c] = 0
        W[(k_offset + 0) * n + 1][c] = 0
        W[(k_offset + 1) * n + 1][c] = 1
        c += 1
        
        # P7
        U[0 * m + (j_offset + 1)][c] = 1 # b
        U[1 * m + (j_offset + 1)][c] = -1 # -d
        V[(j_offset + 0) * p + (k_offset + 0)][c] = 1 # e
        V[(j_offset + 1) * p + (k_offset + 0)][c] = 1 # g
        W[(k_offset + 0) * n + 0][c] = 1
        W[(k_offset + 1) * n + 0][c] = 0
        W[(k_offset + 0) * n + 1][c] = 0
        W[(k_offset + 1) * n + 1][c] = 0
        
    # Block 1: j in {1, 2}
    # k splits: 0-1, 2-3, 4-5
    add_strassen_block(1, 0, 12)
    add_strassen_block(1, 2, 19)
    add_strassen_block(1, 4, 26)
    
    # Block 2: j in {3, 4}
    add_strassen_block(3, 0, 33)
    add_strassen_block(3, 2, 40)
    add_strassen_block(3, 4, 47)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
