import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Define the 7 Strassen products for 2x2 matrices
    # Each product is defined by coefficients for X (2x2) and Y (2x2)
    # and the contribution to Z (2x2)
    
    # Inner Strassen coefficients
    # M1: X11+X22, Y11+Y22 -> Z11+1, Z22+1
    # M2: X21+X22, Y11 -> Z21+1, Z22-1
    # M3: X11, Y12-Y22 -> Z12+1, Z22+1
    # M4: X22, Y21-Y11 -> Z11+1, Z21+1
    # M5: X11+X12, Y22 -> Z11-1, Z12+1
    # M6: X21-X11, Y11+Y12 -> Z22+1
    # M7: X12-X22, Y21+Y22 -> Z11+1
    
    # We need to define the block-level linear combinations
    # A blocks: A11 (0,0-1,1), A12 (0,0-1,2-3), A21 (2,3-0,1), A22 (2,3-2,3)
    # B blocks: B11 (0,1-0,1), B12 (0,1-2,3), B21 (2,3-0,1), B22 (2,3-2,3)
    # C blocks: C11 (0,1-0,1), C12 (0,1-2,3), C21 (2,3-0,1), C22 (2,3-2,3)
    
    # Indices in 2x2 block space:
    # 0: 11, 1: 12, 2: 21, 3: 22
    
    # Block-level Strassen:
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will iterate over the 7 block products s=0..6
    # and the 7 inner products t=0..6
    
    # Helper to map block index to global indices
    # Block 0 (11): rows 0,1; cols 0,1
    # Block 1 (12): rows 0,1; cols 2,3
    # Block 2 (21): rows 2,3; cols 0,1
    # Block 3 (22): rows 2,3; cols 2,3
    
    def get_block_indices(block_id, type_):
        # type_: 'A', 'B', 'C'
        if type_ == 'A':
            if block_id == 0: return [(0,0), (0,1), (1,0), (1,1)]
            if block_id == 1: return [(0,2), (0,3), (1,2), (1,3)]
            if block_id == 2: return [(2,0), (2,1), (3,0), (3,1)]
            if block_id == 3: return [(2,2), (2,3), (3,2), (3,3)]
        elif type_ == 'B':
            if block_id == 0: return [(0,0), (0,1), (1,0), (1,1)]
            if block_id == 1: return [(0,2), (0,3), (1,2), (1,3)]
            if block_id == 2: return [(2,0), (2,1), (3,0), (3,1)]
            if block_id == 3: return [(2,2), (2,3), (3,2), (3,3)]
        elif type_ == 'C':
            if block_id == 0: return [(0,0), (0,1), (1,0), (1,1)]
            if block_id == 1: return [(0,2), (0,3), (1,2), (1,3)]
            if block_id == 2: return [(2,0), (2,1), (3,0), (3,1)]
            if block_id == 3: return [(2,2), (2,3), (3,2), (3,3)]
            
    # Inner Strassen coefficients
    # X coeffs for M_t
    # Y coeffs for M_t
    # Z coeffs for M_t
    
    # M1: X11+X22, Y11+Y22 -> Z11+1, Z22+1
    # M2: X21+X22, Y11 -> Z21+1, Z22-1
    # M3: X11, Y12-Y22 -> Z12+1, Z22+1
    # M4: X22, Y21-Y11 -> Z11+1, Z21+1
    # M5: X11+X12, Y22 -> Z11-1, Z12+1
    # M6: X21-X11, Y11+Y12 -> Z22+1
    # M7: X12-X22, Y21+Y22 -> Z11+1
    
    inner_x = [
        [[1,0],[0,1]], # M1: X11, X22
        [[0,0],[1,1]], # M2: X21, X22
        [[1,0],[0,0]], # M3: X11
        [[0,0],[0,1]], # M4: X22
        [[1,1],[0,0]], # M5: X11, X12
        [[-1,0],[1,0]],# M6: -X11, X21
        [[0,-1],[0,1]] # M7: -X22, X12
    ]
    inner_y = [
        [[1,0],[0,1]], # M1: Y11, Y22
        [[1,0],[0,0]], # M2: Y11
        [[0,1],[0,-1]],# M3: Y12, -Y22
        [[0,-1],[1,0]],# M4: -Y11, Y21
        [[0,0],[0,1]], # M5: Y22
        [[1,1],[0,0]], # M6: Y11, Y12
        [[0,1],[1,1]]  # M7: Y12, Y21, Y22? No. Y21+Y22.
                       # M7: (X12-X22)(Y21+Y22). Y coeffs: Y21, Y22.
                       # Indices: 21 is (1,0) in 2x2? No.
                       # 2x2 indices: 0:11, 1:12, 2:21, 3:22.
                       # Y21 is index 2. Y22 is index 3.
                       # So [[0,0],[1,1]]
    ]
    # Correction for M7 Y coeffs:
    inner_y[6] = [[0,0],[1,1]]
    
    inner_z = [
        [[1,0],[0,1]], # M1: Z11, Z22
        [[0,0],[1,-1]],# M2: Z21, -Z22
        [[0,1],[0,1]], # M3: Z12, Z22
        [[1,0],[1,0]], # M4: Z11, Z21
        [[-1,1],[0,0]],# M5: -Z11, Z12
        [[0,0],[0,1]], # M6: Z22
        [[1,0],[0,0]]  # M7: Z11
    ]
    
    # Block-level Strassen
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # Coeffs for A blocks (2x2 matrix of block indices)
    # Block indices: 0:11, 1:12, 2:21, 3:22
    block_x = [
        [[1,0],[0,1]], # M1: A11, A22
        [[0,0],[1,1]], # M2: A21, A22
        [[1,0],[0,0]], # M3: A11
        [[0,0],[0,1]], # M4: A22
        [[1,1],[0,0]], # M5: A11, A12
        [[-1,0],[1,0]],# M6: -A11, A21
        [[0,-1],[0,1]] # M7: -A22, A12
    ]
    block_y = [
        [[1,0],[0,1]], # M1: B11, B22
        [[1,0],[0,0]], # M2: B11
        [[0,1],[0,-1]],# M3: B12, -B22
        [[0,-1],[1,0]],# M4: -B11, B21
        [[0,0],[0,1]], # M5: B22
        [[1,1],[0,0]], # M6: B11, B12
        [[0,0],[1,1]]  # M7: B21, B22
    ]
    block_z = [
        [[1,0],[0,1]], # M1: C11, C22
        [[0,0],[1,-1]],# M2: C21, -C22
        [[0,1],[0,1]], # M3: C12, C22
        [[1,0],[1,0]], # M4: C11, C21
        [[-1,1],[0,0]],# M5: -C11, C12
        [[0,0],[0,1]], # M6: C22
        [[1,0],[0,0]]  # M7: C11
    ]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            # Combine block coeffs and inner coeffs
            # U[i,j,r] = sum over block a, inner x
            # V[j,k,r] = sum over block b, inner y
            # W[k,i,r] = sum over block c, inner z
            
            # Iterate over all global indices
            for i in range(4):
                for j in range(4):
                    # Determine which block of A this (i,j) belongs to
                    # i: 0,1 -> top, 2,3 -> bottom
                    # j: 0,1 -> left, 2,3 -> right
                    a_row = 0 if i < 2 else 1
                    a_col = 0 if j < 2 else 1
                    a_block = a_row * 2 + a_col
                    
                    # Sum over inner x contributions
                    u_val = 0
                    for bx_row in range(2):
                        for bx_col in range(2):
                            coeff = block_x[s][bx_row][bx_col]
                            if coeff != 0:
                                # This block bx_row, bx_col contributes
                                # The inner X index is bx_row, bx_col
                                inner_coeff = inner_x[t][bx_row][bx_col]
                                if inner_coeff != 0:
                                    # Check if this inner X index corresponds to the local position in the block
                                    # Local position in block:
                                    # i_local = i % 2
                                    # j_local = j % 2
                                    i_local = i % 2
                                    j_local = j % 2
                                    if bx_row == i_local and bx_col == j_local:
                                        u_val += coeff * inner_coeff
                    U[i*4+j, r] = u_val

            for j in range(4):
                for k in range(4):
                    b_row = 0 if j < 2 else 1
                    b_col = 0 if k < 2 else 1
                    b_block = b_row * 2 + b_col
                    
                    v_val = 0
                    for by_row in range(2):
                        for by_col in range(2):
                            coeff = block_y[s][by_row][by_col]
                            if coeff != 0:
                                inner_coeff = inner_y[t][by_row][by_col]
                                if inner_coeff != 0:
                                    j_local = j % 2
                                    k_local = k % 2
                                    if by_row == j_local and by_col == k_local:
                                        v_val += coeff * inner_coeff
                    V[j*4+k, r] = v_val

            for k in range(4):
                for i in range(4):
                    c_row = 0 if i < 2 else 1
                    c_col = 0 if k < 2 else 1
                    c_block = c_row * 2 + c_col
                    
                    w_val = 0
                    for bz_row in range(2):
                        for bz_col in range(2):
                            coeff = block_z[s][bz_row][bz_col]
                            if coeff != 0:
                                inner_coeff = inner_z[t][bz_row][bz_col]
                                if inner_coeff != 0:
                                    i_local = i % 2
                                    k_local = k % 2
                                    if bz_row == i_local and bz_col == k_local:
                                        w_val += coeff * inner_coeff
                    W[k*4+i, r] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
