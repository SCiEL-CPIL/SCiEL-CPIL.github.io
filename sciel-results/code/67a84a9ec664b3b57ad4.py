import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a rank-1 tensor
    def add_tensor(u_indices, v_indices, w_indices, coeff=1):
        nonlocal col
        # u_indices: list of (i, j)
        # v_indices: list of (j, k)
        # w_indices: list of (k, i)
        for (i, j) in u_indices:
            U[i * m + j, col] += coeff
        for (j, k) in v_indices:
            V[j * p + k, col] += coeff
        for (k, i) in w_indices:
            W[k * n + i, col] += coeff
        col += 1

    # Strassen for 2x2 * 2x2
    # Local indices 0,1 for rows/cols of A, B, C
    # Global offsets for A, B, C
    def strassen_2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        # m1 = (A11 + A22)(B11 + B22) -> C11, C22
        # Local A: (0,0), (1,1). B: (0,0), (1,1). C: (0,0), (1,1)
        add_tensor(
            [(a_row_off+0, a_col_off+0), (a_row_off+1, a_col_off+1)],
            [(b_row_off+0, b_col_off+0), (b_row_off+1, b_col_off+1)],
            [(c_col_off+0, c_row_off+0), (c_col_off+1, c_row_off+1)]
        )
        
        # m2 = (A21 + A22)B11 -> C21, C22
        add_tensor(
            [(a_row_off+1, a_col_off+0), (a_row_off+1, a_col_off+1)],
            [(b_row_off+0, b_col_off+0)],
            [(c_col_off+1, c_row_off+1), (c_col_off+1, c_row_off+1)] # Wait, C21 and C22?
            # z21 = m2 + m4. z22 = m1 - m2 + m3 + m6.
            # So m2 contributes to C21 and C22?
            # z21 (row 1, col 0) and z22 (row 1, col 1).
            # W indices are (k, i).
            # C21 is i=1, k=0 -> W[0*2+1] = W[1].
            # C22 is i=1, k=1 -> W[1*2+1] = W[3].
            # So w_indices should be [(0, 1), (1, 1)]?
            # My add_tensor takes (k, i).
            # So for C21 (i=1, k=0), index is (0, 1).
            # For C22 (i=1, k=1), index is (1, 1).
        )
        # I need to be careful with Strassen formulas.
        
    # Let's just hardcode the 7 Strassen terms for a generic 2x2 block
    # with given global offsets.
    
    def strassen_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        # A is at rows a_i0..a_i0+1, cols a_j0..a_j0+1
        # B is at rows b_j0..b_j0+1, cols b_k0..b_k0+1
        # C is at rows c_i0..c_i0+1, cols c_k0..c_k0+1
        
        # m1 = (A11 + A22)(B11 + B22)
        # A11: (a_i0, a_j0), A22: (a_i0+1, a_j0+1)
        # B11: (b_j0, b_k0), B22: (b_j0+1, b_k0+1)
        # C11: (c_i0, c_k0), C22: (c_i0+1, c_k0+1)
        add_tensor(
            [(a_i0, a_j0), (a_i0+1, a_j0+1)],
            [(b_j0, b_k0), (b_j0+1, b_k0+1)],
            [(c_k0, c_i0), (c_k0+1, c_i0+1)]
        )
        
        # m2 = (A21 + A22)B11
        # A21: (a_i0+1, a_j0), A22: (a_i0+1, a_j0+1)
        # B11: (b_j0, b_k0)
        # C21: (c_i0+1, c_k0), C22: (c_i0+1, c_k0+1)
        # z21 += m2, z22 -= m2
        add_tensor(
            [(a_i0+1, a_j0), (a_i0+1, a_j0+1)],
            [(b_j0, b_k0)],
            [(c_k0, c_i0+1), (c_k0+1, c_i0+1)] # Coeff 1 for C21, -1 for C22?
        )
        # Wait, add_tensor adds coeff * 1. I need to handle signs.
        # I will modify add_tensor to accept a list of (indices, coeff) or just pass signs in indices?
        # Better: add_tensor takes u, v, w, and a sign for each part?
        # No, the rank-1 tensor has fixed coefficients in U, V, W.
        # The signs are in the combination of the rank-1 tensors to form the final C.
        # But here we are building the tensor T directly.
        # T = sum u_r v_r w_r.
        # The coefficients in U, V, W are the entries of u_r, v_r, w_r.
        # So if m2 contributes -1 to C22, then w_2 has -1 at C22.
        
        # Let's rewrite add_tensor to handle coefficients in the lists.
        
    # Redefine add_tensor
    def add_tensor(u_list, v_list, w_list):
        nonlocal col
        for (idx, coeff) in u_list:
            U[idx, col] += coeff
        for (idx, coeff) in v_list:
            V[idx, col] += coeff
        for (idx, coeff) in w_list:
            W[idx, col] += coeff
        col += 1

    def strassen_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        # m1
        add_tensor(
            [((a_i0)*m + a_j0, 1), ((a_i0+1)*m + a_j0+1, 1)],
            [((b_j0)*p + b_k0, 1), ((b_j0+1)*p + b_k0+1, 1)],
            [((c_k0)*n + c_i0, 1), ((c_k0+1)*n + c_i0+1, 1)]
        )
        
        # m2
        # C21 coeff 1, C22 coeff -1
        add_tensor(
            [((a_i0+1)*m + a_j0, 1), ((a_i0+1)*m + a_j0+1, 1)],
            [((b_j0)*p + b_k0, 1)],
            [((c_k0)*n + c_i0+1, 1), ((c_k0+1)*n + c_i0+1, -1)]
        )
        
        # m3 = A11(B12 - B22)
        # C12 coeff 1
        add_tensor(
            [((a_i0)*m + a_j0, 1)],
            [((b_j0)*p + b_k0+1, 1), ((b_j0+1)*p + b_k0+1, -1)],
            [((c_k0+1)*n + c_i0, 1)]
        )
        
        # m4 = A22(B21 - B11)
        # C21 coeff 1, C11 coeff 1
        add_tensor(
            [((a_i0+1)*m + a_j0+1, 1)],
            [((b_j0+1)*p + b_k0, 1), ((b_j0)*p + b_k0, -1)],
            [((c_k0)*n + c_i0+1, 1), ((c_k0)*n + c_i0, 1)]
        )
        
        # m5 = (A11 + A12)B22
        # C11 coeff -1, C12 coeff 1
        add_tensor(
            [((a_i0)*m + a_j0, 1), ((a_i0)*m + a_j0+1, 1)],
            [((b_j0+1)*p + b_k0+1, 1)],
            [((c_k0)*n + c_i0, -1), ((c_k0+1)*n + c_i0, 1)]
        )
        
        # m6 = (A21 - A11)(B11 + B12)
        # C22 coeff 1
        add_tensor(
            [((a_i0+1)*m + a_j0, 1), ((a_i0)*m + a_j0, -1)],
            [((b_j0)*p + b_k0, 1), ((b_j0)*p + b_k0+1, 1)],
            [((c_k0+1)*n + c_i0+1, 1)]
        )
        
        # m7 = (A12 - A22)(B21 + B22)
        # C11 coeff 1, C22 coeff 1
        add_tensor(
            [((a_i0)*m + a_j0+1, 1), ((a_i0+1)*m + a_j0+1, -1)],
            [((b_j0+1)*p + b_k0, 1), ((b_j0+1)*p + b_k0+1, 1)],
            [((c_k0)*n + c_i0, 1), ((c_k0+1)*n + c_i0+1, 1)]
        )

    # Block 1: A(0:2, 0:2), B(0:2, 0:2), C(0:2, 0:2)
    strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A(0:2, 0:2), B(0:2, 2:4), C(0:2, 2:4)
    strassen_block(0, 0, 0, 2, 0, 2)
    
    # Block 3: A(0:2, 0:2), B(0:2, 4:6), C(0:2, 4:6)
    strassen_block(0, 0, 0, 4, 0, 4)
    
    # Block 4: A(0:2, 0:2), B(0:2, 6:7), C(0:2, 6:7)
    # Schoolbook 2x2 * 2x1
    # C_i6 = A_i0 B_06 + A_i1 B_16
    # i=0: C_06 = A_00 B_06 + A_01 B_16
    # i=1: C_16 = A_10 B_06 + A_11 B_16
    
    # Term 1: A_00 B_06 -> C_06
    add_tensor(
        [((0)*m + 0, 1)],
        [((0)*p + 6, 1)],
        [((6)*n + 0, 1)]
    )
    # Term 2: A_10 B_06 -> C_16
    add_tensor(
        [((1)*m + 0, 1)],
        [((0)*p + 6, 1)],
        [((6)*n + 1, 1)]
    )
    # Term 3: A_01 B_16 -> C_06
    add_tensor(
        [((0)*m + 1, 1)],
        [((1)*p + 6, 1)],
        [((6)*n + 0, 1)]
    )
    # Term 4: A_11 B_16 -> C_16
    add_tensor(
        [((1)*m + 1, 1)],
        [((1)*p + 6, 1)],
        [((6)*n + 1, 1)]
    )
    
    # Block 5: A(0:2, 2:3), B(2:3, 0:7), C(0:2, 0:7)
    # A is 2x1 (col 2). B is 1x7 (row 2).
    # C_i,k = A_i,2 B_2,k
    for i in range(2):
        for k in range(7):
            add_tensor(
                [((i)*m + 2, 1)],
                [((2)*p + k, 1)],
                [((k)*n + i, 1)]
            )
            
    return {"U": U, "V": V, "W": W}
