"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on the inner dimension split.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Helper to add a Strassen block
    def add_strassen_block(col_offset, a_idx, b_idx, c_idx):
        # a_idx: [A11, A12, A21, A22] indices in U
        # b_idx: [B11, B12, B21, B22] indices in V
        # c_idx: [C11, C12, C21, C22] indices in W
        
        # M1
        col = col_offset + 0
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        
        # M2
        col = col_offset + 1
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        
        # M3
        col = col_offset + 2
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[1]][col] += 1
        W[c_idx[2]][col] += 1
        
        # M4
        col = col_offset + 3
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        
        # M5
        col = col_offset + 4
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        
        # M6
        col = col_offset + 5
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        
        # M7
        col = col_offset + 6
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1

    # Part 1: A_L * B_T
    # A_L is 2x2, indices in U:
    # A11=0, A12=1, A21=2, A22=3
    a_idx_L = [0, 1, 2, 3]
    
    # B_T is 2x8, split into 4 blocks of 2x2
    for j in range(4):
        # B block j indices in V:
        # B11 = 0 + 2j, B12 = 1 + 2j, B21 = 2 + 2j, B22 = 3 + 2j
        b_idx = [2*j, 2*j+1, 2*j+2, 2*j+3]
        
        # C block j indices in W:
        # C11 (row 0, col 2j) -> idx 4j
        # C12 (row 0, col 2j+1) -> idx 4j+2
        # C21 (row 1, col 2j) -> idx 4j+1
        # C22 (row 1, col 2j+1) -> idx 4j+3
        c_idx = [4*j, 4*j+2, 4*j+1, 4*j+3]
        
        add_strassen_block(j*7, a_idx_L, b_idx, c_idx)

    # Part 2: A_R * B_B
    # A_R is 2x2, indices in U:
    # A11=4, A12=5, A21=6, A22=7
    a_idx_R = [4, 5, 6, 7]
    
    # B_B is 2x8, split into 4 blocks of 2x2
    # B_B starts at row 2 of B, so V index offset is 8 (2*4)
    for j in range(4):
        # B block j indices in V:
        # B11 = 8 + 2j, B12 = 9 + 2j, B21 = 10 + 2j, B22 = 11 + 2j
        b_idx = [8 + 2*j, 8 + 2*j+1, 8 + 2*j+2, 8 + 2*j+3]
        
        # C block j indices in W are the same as Part 1
        c_idx = [4*j, 4*j+2, 4*j+1, 4*j+3]
        
        add_strassen_block(28 + j*7, a_idx_R, b_idx, c_idx)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
