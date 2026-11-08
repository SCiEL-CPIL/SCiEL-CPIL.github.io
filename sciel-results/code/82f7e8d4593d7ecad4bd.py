"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on 2x2x2 blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # a_idx: [00, 01, 10, 11] indices into U
    # b_idx: [00, 01, 10, 11] indices into V
    # w_idx: [00, 01, 10, 11] indices into W (corresponding to C_00, C_01, C_10, C_11)
    # col_offset: starting column in U, V, W
    def add_strassen(a_idx, b_idx, w_idx, col_offset):
        # M1: (a00+a11)(b00+b11)
        U[a_idx[0]][col_offset] += 1
        U[a_idx[3]][col_offset] += 1
        V[b_idx[0]][col_offset] += 1
        V[b_idx[3]][col_offset] += 1
        W[w_idx[0]][col_offset] += 1
        W[w_idx[3]][col_offset] += 1
        
        # M2: (a10+a11)b00
        U[a_idx[2]][col_offset+1] += 1
        U[a_idx[3]][col_offset+1] += 1
        V[b_idx[0]][col_offset+1] += 1
        W[w_idx[2]][col_offset+1] += 1
        W[w_idx[3]][col_offset+1] -= 1
        
        # M3: a00(b01-b11)
        U[a_idx[0]][col_offset+2] += 1
        V[b_idx[1]][col_offset+2] += 1
        V[b_idx[3]][col_offset+2] -= 1
        W[w_idx[1]][col_offset+2] += 1
        W[w_idx[3]][col_offset+2] += 1
        
        # M4: a11(b10-b00)
        U[a_idx[3]][col_offset+3] += 1
        V[b_idx[2]][col_offset+3] += 1
        V[b_idx[0]][col_offset+3] -= 1
        W[w_idx[0]][col_offset+3] += 1
        W[w_idx[2]][col_offset+3] += 1
        
        # M5: (a00+a01)b11
        U[a_idx[0]][col_offset+4] += 1
        U[a_idx[1]][col_offset+4] += 1
        V[b_idx[3]][col_offset+4] += 1
        W[w_idx[0]][col_offset+4] -= 1
        W[w_idx[1]][col_offset+4] += 1
        
        # M6: (a10-a00)(b00+b01)
        U[a_idx[2]][col_offset+5] += 1
        U[a_idx[0]][col_offset+5] -= 1
        V[b_idx[0]][col_offset+5] += 1
        V[b_idx[1]][col_offset+5] += 1
        W[w_idx[3]][col_offset+5] += 1
        
        # M7: (a01-a11)(b10+b11)
        U[a_idx[1]][col_offset+6] += 1
        U[a_idx[3]][col_offset+6] -= 1
        V[b_idx[2]][col_offset+6] += 1
        V[b_idx[3]][col_offset+6] += 1
        W[w_idx[0]][col_offset+6] += 1
        W[w_idx[3]][col_offset+6] += 1

    # Indices for A (2x4)
    # A[i, j] -> i*4 + j
    a_L = [0, 1, 4, 5]    # A[0,0], A[0,1], A[1,0], A[1,1]
    a_R = [2, 3, 6, 7]    # A[0,2], A[0,3], A[1,2], A[1,3]
    
    # Indices for B (4x4)
    # B[j, k] -> j*4 + k
    b_T0 = [0, 1, 4, 5]   # B[0,0], B[0,1], B[1,0], B[1,1]
    b_T1 = [2, 3, 6, 7]   # B[0,2], B[0,3], B[1,2], B[1,3]
    b_B0 = [8, 9, 12, 13] # B[2,0], B[2,1], B[3,0], B[3,1]
    b_B1 = [10, 11, 14, 15] # B[2,2], B[2,3], B[3,2], B[3,3]
    
    # Indices for W (4x2 output, indexed by k*n+i)
    # C[i, k] -> k*2 + i
    # k in {0,1}
    w_L0 = [0, 2, 1, 3]   # C[0,0], C[0,1], C[1,0], C[1,1]
    # k in {2,3}
    w_L1 = [4, 6, 5, 7]   # C[0,2], C[0,3], C[1,2], C[1,3]
    
    # Block 1: A_L * B_T0
    add_strassen(a_L, b_T0, w_L0, 0)
    # Block 2: A_L * B_T1
    add_strassen(a_L, b_T1, w_L1, 7)
    # Block 3: A_R * B_B0
    add_strassen(a_R, b_B0, w_L0, 14)
    # Block 4: A_R * B_B1
    add_strassen(a_R, b_B1, w_L1, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
