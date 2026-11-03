"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set Strassen block
    # a_idx, b_idx, c_idx are local indices for A, B, C
    # We map them to global indices via offsets
    def add_strassen(u_row_offsets, v_row_offsets, w_row_offsets, col_start):
        # U factors
        u_factors = [
            {0: 1, 1: 1},  # a+d
            {2: 1, 3: 1},  # c+d
            {0: 1},        # a
            {3: 1},        # d
            {0: 1, 1: 1},  # a+b
            {2: 1, 0: -1}, # c-a
            {1: 1, 3: -1}  # b-d
        ]
        # V factors
        v_factors = [
            {0: 1, 1: 1},  # e+h
            {0: 1},        # e
            {2: 1, 3: -1}, # f-h
            {2: 1, 0: -1}, # g-e
            {3: 1},        # h
            {0: 1, 2: 1},  # e+f
            {2: 1, 3: 1}   # g+h
        ]
        # W factors (coeffs for C)
        w_factors = [
            {0: 1, 3: 1},  # C11+1, C22+1
            {2: 1, 3: -1}, # C21+1, C22-1
            {1: 1, 3: 1},  # C12+1, C22+1
            {0: 1, 2: 1},  # C11+1, C21+1
            {0: -1, 1: 1}, # C11-1, C12+1
            {3: 1},        # C22+1
            {0: 1}         # C11+1
        ]
        
        for r in range(7):
            col = col_start + r
            for local_i, val in u_factors[r].items():
                U[u_row_offsets[local_i]][col] = val
            for local_j, val in v_factors[r].items():
                V[v_row_offsets[local_j]][col] = val
            for local_k, val in w_factors[r].items():
                W[w_row_offsets[local_k]][col] = val

    # Block 1: A[:, 0:2] * B[0:2, 0:2]
    # A local: 0,1,2,3 -> Global: i*3+j
    u_off_1 = [0, 1, 3, 4] # a,b,c,d -> (0,0), (0,1), (1,0), (1,1)
    # B local: 0,1,2,3 -> Global: j*5+k
    v_off_1 = [0, 2, 5, 7] # e,f,g,h -> (0,0), (0,1), (1,0), (1,1)
    # C local: 0,1,2,3 -> Global: k*2+i
    w_off_1 = [0, 1, 2, 3] # C11,C12,C21,C22 -> (0,0), (0,1), (1,0), (1,1)
    add_strassen(u_off_1, v_off_1, w_off_1, 0)
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4]
    # A same
    # B local: 0,1,2,3 -> Global: j*5+k (k=2,3)
    v_off_2 = [2, 3, 7, 8] # e,f,g,h -> (0,2), (0,3), (1,2), (1,3)
    # C local: 0,1,2,3 -> Global: k*2+i (k=2,3)
    w_off_2 = [4, 5, 6, 7] # C11,C12,C21,C22 -> (2,0), (2,1), (3,0), (3,1)
    add_strassen(u_off_1, v_off_2, w_off_2, 7)
    
    # Block 3: A[:, 0:2] * B[0:2, 4:5]
    # A same
    # B local: 0,1 -> Global: j*5+k (k=4)
    # C local: 0,1 -> Global: k*2+i (k=4)
    # Schoolbook 2x2x1
    # Col 14: A00*B00
    U[0][14] = 1
    V[4][14] = 1
    W[8][14] = 1
    # Col 15: A01*B10
    U[1][15] = 1
    V[9][15] = 1
    W[8][15] = 1
    # Col 16: A10*B00
    U[3][16] = 1
    V[4][16] = 1
    W[9][16] = 1
    # Col 17: A11*B10
    U[4][17] = 1
    V[9][17] = 1
    W[9][17] = 1
    
    # Block 4: A[:, 2] * B[2, :]
    # Outer product
    col = 18
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
