import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Strassen terms definitions
    # Each term is (x_coeffs, y_coeffs, z_coeffs)
    # x_coeffs: dict mapping local x index to coeff
    # y_coeffs: dict mapping local y index to coeff
    # z_coeffs: dict mapping local z index to coeff
    # Local indices: 11, 12, 21, 22
    
    # M1: (x11 + x22)(y11 + y22)
    # z11 += 1, z22 += 1
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1: x11,x22; y11,y22; z11,z22
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),             # M2: (x21+x22)y11 -> z21 += 1
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),            # M3: x11(y12-y22) -> z12 += 1
        ({3: 1}, {2: 1, 0: -1}, {2: 1, 3: 1}),      # M4: x22(y21-y11) -> z21 += 1, z22 += 1
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),      # M5: (x11+x12)y22 -> z12 += 1, z11 -= 1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # M6: (x21-x11)(y11+y12) -> z22 += 1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})       # M7: (x12-x22)(y21+y22) -> z11 += 1
    ]
    
    # Note: I need to be careful with the mapping of local indices to global indices
    # and the signs in the z formulas.
    
    # Let's re-verify the Strassen formulas and signs.
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    # Let's update the z_coeffs in the list above based on this.
    # Indices: 0=11, 1=12, 2=21, 3=22
    
    terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1}), # z21 gets M2
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # z12 gets M3, z22 gets M3
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # z11 gets M4, z21 gets M4
        # M5
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}), # z12 gets M5, z11 gets -M5
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # z22 gets M6
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})  # z11 gets M7
    ]
    
    # Block definitions
    # (x_global_indices, y_global_indices, z_global_indices)
    # x_global_indices: [x11, x12, x21, x22]
    # y_global_indices: [y11, y12, y21, y22]
    # z_global_indices: [z11, z12, z21, z22]
    
    blocks = []
    
    # Block 1: A_left * B_top_left -> C_top_left (cols 0,1)
    # A_left: A[0,0], A[0,1], A[1,0], A[1,1] -> 0, 1, 4, 5
    # B_top_left: B[0,0], B[0,1], B[1,0], B[1,1] -> 0, 1, 4, 5
    # C_top_left: C[0,0], C[0,1], C[1,0], C[1,1] -> 0, 1, 2, 3
    blocks.append(([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3]))
    
    # Block 2: A_right * B_bottom_left -> C_bottom_left? No, C[:, 0:2]
    # A_right: A[0,2], A[0,3], A[1,2], A[1,3] -> 2, 3, 6, 7
    # B_bottom_left: B[2,0], B[2,1], B[3,0], B[3,1] -> 8, 9, 12, 13
    # C[:, 0:2] is same as Block 1's output indices: 0, 1, 2, 3
    blocks.append(([2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 2, 3]))
    
    # Block 3: A_left * B_top_right -> C[:, 2:4]
    # A_left: 0, 1, 4, 5
    # B_top_right: B[0,2], B[0,3], B[1,2], B[1,3] -> 2, 3, 6, 7
    # C[:, 2:4]: C[0,2], C[0,3], C[1,2], C[1,3] -> 2, 3, 4, 5
    blocks.append(([0, 1, 4, 5], [2, 3, 6, 7], [2, 3, 4, 5]))
    
    # Block 4: A_right * B_bottom_right -> C[:, 2:4]
    # A_right: 2, 3, 6, 7
    # B_bottom_right: B[2,2], B[2,3], B[3,2], B[3,3] -> 10, 11, 14, 15
    # C[:, 2:4]: 2, 3, 4, 5
    blocks.append(([2, 3, 6, 7], [10, 11, 14, 15], [2, 3, 4, 5]))
    
    col_idx = 0
    for x_idx, y_idx, z_idx in blocks:
        for x_coeffs, y_coeffs, z_coeffs in terms:
            # Fill U
            for local_i, coeff in x_coeffs.items():
                global_i = x_idx[local_i]
                U[global_i][col_idx] += coeff
                
            # Fill V
            for local_j, coeff in y_coeffs.items():
                global_j = y_idx[local_j]
                V[global_j][col_idx] += coeff
                
            # Fill W
            for local_k, coeff in z_coeffs.items():
                global_k = z_idx[local_k]
                W[global_k][col_idx] += coeff
                
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
