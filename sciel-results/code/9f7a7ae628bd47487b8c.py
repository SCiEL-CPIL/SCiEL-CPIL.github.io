import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a rank-28 decomposition based on 4 independent 2x2x2 Strassen multiplications.
    # A (2x4) is split into A_L (2x2) and A_R (2x2).
    # B (4x4) is split into B_11, B_12, B_21, B_22 (all 2x2).
    # C_L = A_L B_11 + A_R B_21
    # C_R = A_L B_12 + A_R B_22
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U indices: i*m + j. A is 2x4.
    # A_L: i in 0..1, j in 0..1. Index i*4 + j
    # A_R: i in 0..1, j in 2..3. Index i*4 + (j+2)
    
    # V indices: j*p + k. B is 4x4.
    # B_11: j in 0..1, k in 0..1. Index j*4 + k
    # B_12: j in 0..1, k in 2..3. Index j*4 + (k+2)
    # B_21: j in 2..3, k in 0..1. Index (j+2)*4 + k
    # B_22: j in 2..3, k in 2..3. Index (j+2)*4 + (k+2)
    
    # W indices: k*n + i. C is 4x2.
    # C_L: k in 0..1, i in 0..1. Index k*2 + i
    # C_R: k in 2..3, i in 0..1. Index (k+2)*2 + i
    
    def add_strassen_block(col_offset, A_type, B_type, C_type):
        # A_type: 'L' or 'R'
        # B_type: '11', '12', '21', '22'
        # C_type: 'L' or 'R'
        
        # Helper to get A index
        def get_A_idx(i, j_local):
            if A_type == 'L':
                return i * 4 + j_local
            else:
                return i * 4 + (j_local + 2)
        
        # Helper to get B index
        def get_B_idx(j_local, k_local):
            if B_type == '11':
                return j_local * 4 + k_local
            elif B_type == '12':
                return j_local * 4 + (k_local + 2)
            elif B_type == '21':
                return (j_local + 2) * 4 + k_local
            else: # 22
                return (j_local + 2) * 4 + (k_local + 2)
        
        # Helper to get C index
        def get_C_idx(k_local, i):
            if C_type == 'L':
                return k_local * 2 + i
            else:
                return (k_local + 2) * 2 + i

        # Strassen steps
        # M1
        c = col_offset + 0
        U[get_A_idx(0, 0)][c] += 1
        U[get_A_idx(1, 1)][c] += 1
        V[get_B_idx(0, 0)][c] += 1
        V[get_B_idx(1, 1)][c] += 1
        W[get_C_idx(0, 0)][c] += 1
        W[get_C_idx(0, 1)][c] += 0 # C00 = M1...
        W[get_C_idx(1, 0)][c] += 0
        W[get_C_idx(1, 1)][c] += 1
        
        # M2
        c = col_offset + 1
        U[get_A_idx(1, 0)][c] += 1
        U[get_A_idx(1, 1)][c] += 1
        V[get_B_idx(0, 0)][c] += 1
        W[get_C_idx(1, 0)][c] += 1
        W[get_C_idx(1, 1)][c] += -1
        
        # M3
        c = col_offset + 2
        U[get_A_idx(0, 0)][c] += 1
        V[get_B_idx(0, 1)][c] += 1
        V[get_B_idx(1, 1)][c] += -1
        W[get_C_idx(0, 1)][c] += 1
        W[get_C_idx(1, 1)][c] += 1
        
        # M4
        c = col_offset + 3
        U[get_A_idx(1, 1)][c] += 1
        V[get_B_idx(1, 0)][c] += 1
        V[get_B_idx(0, 0)][c] += -1
        W[get_C_idx(0, 0)][c] += 1
        W[get_C_idx(1, 0)][c] += 1
        
        # M5
        c = col_offset + 4
        U[get_A_idx(0, 0)][c] += 1
        U[get_A_idx(0, 1)][c] += 1
        V[get_B_idx(1, 1)][c] += 1
        W[get_C_idx(0, 0)][c] += -1
        W[get_C_idx(1, 1)][c] += 1
        
        # M6
        c = col_offset + 5
        U[get_A_idx(1, 0)][c] += 1
        U[get_A_idx(0, 0)][c] += -1
        V[get_B_idx(0, 0)][c] += 1
        V[get_B_idx(0, 1)][c] += 1
        W[get_C_idx(1, 1)][c] += 1
        
        # M7
        c = col_offset + 6
        U[get_A_idx(0, 1)][c] += 1
        U[get_A_idx(1, 1)][c] += -1
        V[get_B_idx(1, 0)][c] += 1
        V[get_B_idx(1, 1)][c] += 1
        W[get_C_idx(0, 0)][c] += 1
        W[get_C_idx(0, 1)][c] += -1

    # Block 1: A_L * B_11 -> C_L
    add_strassen_block(0, 'L', '11', 'L')
    # Block 2: A_L * B_12 -> C_R
    add_strassen_block(7, 'L', '12', 'R')
    # Block 3: A_R * B_21 -> C_L
    add_strassen_block(14, 'R', '21', 'L')
    # Block 4: A_R * B_22 -> C_R
    add_strassen_block(21, 'R', '22', 'R')
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
