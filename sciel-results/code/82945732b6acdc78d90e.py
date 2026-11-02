"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    - Use Strassen's algorithm for the two 4x4x4 blocks (columns 0-3 and 4-7 of B), rank 49 each
    - Use schoolbook for the 4x4x1 block (column 8 of B), rank 16
    Total rank: 49 + 49 + 16 = 114
    
    The key insight is that Strassen's algorithm for 2x2 matrices gives rank 7,
    and for 4x4 matrices (viewed as 2x2 blocks of 2x2 matrices) gives rank 49.
    This is achieved by applying the 7 Strassen products at the block level,
    where each block product is itself a 2x2 * 2x2 multiplication that can be
    decomposed into 7 rank-1 terms.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * rank for _ in range(p * n)]  # 36 x 114
    
    # Define the 7 Strassen rank-1 terms for 2x2 matrix multiplication
    # Each term is a tuple of (a_indices, b_indices, c_indices) with signs
    # For 2x2: A[i,j], B[j,k], C[i,k] where i,j,k in {0,1}
    # Using flat indices: A: 0,1,2,3 = (0,0),(0,1),(1,0),(1,1)
    # B: 0,1,2,3 = (0,0),(0,1),(1,0),(1,1)
    # C: 0,1,2,3 = (0,0),(0,1),(1,0),(1,1)
    
    # Strassen 2x2x2 decomposition (7 terms):
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),                # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),               # M3
        ({3: 1}, {2: 1, 0: -1}, {2: 1}),               # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),                # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),         # M7
    ]
    
    def add_4x4_strassen_block(col_offset, rank_offset):
        """
        Add a 4x4x4 Strassen decomposition for B columns [col_offset, col_offset+4).
        Uses 49 rank-1 terms.
        """
        # Block decomposition of 4x4 into 2x2 blocks of 2x2 matrices:
        # A blocks:
        A00 = [0, 1, 2, 3]      # rows 0-1, cols 0-1
        A01 = [4, 5, 6, 7]      # rows 0-1, cols 2-3
        A10 = [8, 9, 10, 11]    # rows 2-3, cols 0-1
        A11 = [12, 13, 14, 15]  # rows 2-3, cols 2-3
        
        # B blocks (with col_offset):
        B00 = [0 * p + col_offset, 0 * p + col_offset + 1, 
               1 * p + col_offset, 1 * p + col_offset + 1]
        B01 = [0 * p + col_offset + 2, 0 * p + col_offset + 3, 
               1 * p + col_offset + 2, 1 * p + col_offset + 3]
        B10 = [2 * p + col_offset, 2 * p + col_offset + 1, 
               3 * p + col_offset, 3 * p + col_offset + 1]
        B11 = [2 * p + col_offset + 2, 2 * p + col_offset + 3, 
               3 * p + col_offset + 2, 3 * p + col_offset + 3]
        
        # C blocks:
        C00 = [0, 1, 2, 3]      # rows 0-1, cols 0-1
        C01 = [4, 5, 6, 7]      # rows 0-1, cols 2-3
        C10 = [8, 9, 10, 11]    # rows 2-3, cols 0-1
        C11 = [12, 13, 14, 15]  # rows 2-3, cols 2-3
        
        # The 7 Strassen products at the block level:
        # P1 = (A00 + A11)(B00 + B11) -> contributes to C00 (+1) and C11 (+1)
        # P2 = (A10 + A11)B00 -> contributes to C10 (+1)
        # P3 = A00(B01 - B11) -> contributes to C01 (+1)
        # P4 = A11(B10 - B00) -> contributes to C10 (+1)
        # P5 = (A00 + A01)B11 -> contributes to C01 (+1)
        # P6 = (A10 - A00)(B00 + B01) -> contributes to C11 (+1)
        # P7 = (A01 - A11)(B10 + B11) -> contributes to C00 (+1)
        
        # For each block product Pi = Xi * Yi where Xi and Yi are 2x2 matrices,
        # we use the 7 Strassen terms for 2x2x2.
        
        # Define the linear combinations for each block product
        # Each entry is a dict {flat_idx: coeff}
        
        # P1: X = A00 + A11, Y = B00 + B11
        P1_X = [
            {A00[0]: 1, A11[0]: 1},  # X00
            {A00[1]: 1, A11[1]: 1},  # X01
            {A00[2]: 1, A11[2]: 1},  # X10
            {A00[3]: 1, A11[3]: 1},  # X11
        ]
        P1_Y = [
            {B00[0]: 1, B11[0]: 1},  # Y00
            {B00[1]: 1, B11[1]: 1},  # Y01
            {B00[2]: 1, B11[2]: 1},  # Y10
            {B00[3]: 1, B11[3]: 1},  # Y11
        ]
        P1_Z = [
            {C00[0]: 1, C11[0]: 1},  # Z00
            {C00[1]: 1, C11[1]: 1},  # Z01
            {C00[2]: 1, C11[2]: 1},  # Z10
            {C00[3]: 1, C11[3]: 1},  # Z11
        ]
        
        # P2: X = A10 + A11, Y = B00
        P2_X = [
            {A10[0]: 1, A11[0]: 1},
            {A10[1]: 1, A11[1]: 1},
            {A10[2]: 1, A11[2]: 1},
            {A10[3]: 1, A11[3]: 1},
        ]
        P2_Y = [
            {B00[0]: 1},
            {B00[1]: 1},
            {B00[2]: 1},
            {B00[3]: 1},
        ]
        P2_Z = [
            {C10[0]: 1},
            {C10[1]: 1},
            {C10[2]: 1},
            {C10[3]: 1},
        ]
        
        # P3: X = A00, Y = B01 - B11
        P3_X = [
            {A00[0]: 1},
            {A00[1]: 1},
            {A00[2]: 1},
            {A00[3]: 1},
        ]
        P3_Y = [
            {B01[0]: 1, B11[0]: -1},
            {B01[1]: 1, B11[1]: -1},
            {B01[2]: 1, B11[2]: -1},
            {B01[3]: 1, B11[3]: -1},
        ]
        P3_Z = [
            {C01[0]: 1},
            {C01[1]: 1},
            {C01[2]: 1},
            {C01[3]: 1},
        ]
        
        # P4: X = A11, Y = B10 - B00
        P4_X = [
            {A11[0]: 1},
            {A11[1]: 1},
            {A11[2]: 1},
            {A11[3]: 1},
        ]
        P4_Y = [
            {B10[0]: 1, B00[0]: -1},
            {B10[1]: 1, B00[1]: -1},
            {B10[2]: 1, B00[2]: -1},
            {B10[3]: 1, B00[3]: -1},
        ]
        P4_Z = [
            {C10[0]: 1},
            {C10[1]: 1},
            {C10[2]: 1},
            {C10[3]: 1},
        ]
        
        # P5: X = A00 + A01, Y = B11
        P5_X = [
            {A00[0]: 1, A01[0]: 1},
            {A00[1]: 1, A01[1]: 1},
            {A00[2]: 1, A01[2]: 1},
            {A00[3]: 1, A01[3]: 1},
        ]
        P5_Y = [
            {B11[0]: 1},
            {B11[1]: 1},
            {B11[2]: 1},
            {B11[3]: 1},
        ]
        P5_Z = [
            {C01[0]: 1},
            {C01[1]: 1},
            {C01[2]: 1},
            {C01[3]: 1},
        ]
        
        # P6: X = A10 - A00, Y = B00 + B01
        P6_X = [
            {A10[0]: 1, A00[0]: -1},
            {A10[1]: 1, A00[1]: -1},
            {A10[2]: 1, A00[2]: -1},
            {A10[3]: 1, A00[3]: -1},
        ]
        P6_Y = [
            {B00[0]: 1, B01[0]: 1},
            {B00[1]: 1, B01[1]: 1},
            {B00[2]: 1, B01[2]: 1},
            {B00[3]: 1, B01[3]: 1},
        ]
        P6_Z = [
            {C11[0]: 1},
            {C11[1]: 1},
            {C11[2]: 1},
            {C11[3]: 1},
        ]
        
        # P7: X = A01 - A11, Y = B10 + B11
        P7_X = [
            {A01[0]: 1, A11[0]: -1},
            {A01[1]: 1, A11[1]: -1},
            {A01[2]: 1, A11[2]: -1},
            {A01[3]: 1, A11[3]: -1},
        ]
        P7_Y = [
            {B10[0]: 1, B11[0]: 1},
            {B10[1]: 1, B11[1]: 1},
            {B10[2]: 1, B11[2]: 1},
            {B10[3]: 1, B11[3]: 1},
        ]
        P7_Z = [
            {C00[0]: 1},
            {C00[1]: 1},
            {C00[2]: 1},
            {C00[3]: 1},
        ]
        
        block_products = [
            (P1_X, P1_Y, P1_Z),
            (P2_X, P2_Y, P2_Z),
            (P3_X, P3_Y, P3_Z),
            (P4_X, P4_Y, P4_Z),
            (P5_X, P5_Y, P5_Z),
            (P6_X, P6_Y, P6_Z),
            (P7_X, P7_Y, P7_Z),
        ]
        
        # For each of the 7 block products, apply 2x2 Strassen
        for bp_idx, (X_defs, Y_defs, Z_defs) in enumerate(block_products):
            for s in range(7):
                a_dict, b_dict, c_dict = strassen_2x2[s]
                r = rank_offset + bp_idx * 7 + s
                
                # Process U entries
                for x_idx, x_coeff in a_dict.items():
                    for a_idx, a_coeff in X_defs[x_idx].items():
                        U[a_idx][r] += x_coeff * a_coeff
                
                # Process V entries
                for y_idx, y_coeff in b_dict.items():
                    for b_idx, b_coeff in Y_defs[y_idx].items():
                        V[b_idx][r] += y_coeff * b_coeff
                
                # Process W entries
                for z_idx, z_coeff in c_dict.items():
                    for c_idx, c_coeff in Z_defs[z_idx].items():
                        W[c_idx][r] += z_coeff * c_coeff
    
    # Add Strassen blocks for columns 0-3 and 4-7
    add_4x4_strassen_block(0, 0)
    add_4x4_strassen_block(4, 49)
    
    # Add schoolbook block for column 8
    rank_offset = 98
    for i in range(n):
        for j in range(m):
            for k in range(1):  # only column 8
                col = rank_offset + i * m * 1 + j * 1 + k
                U[i * m + j][col] = 1
                V[j * p + 8][col] = 1
                W[8 * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
