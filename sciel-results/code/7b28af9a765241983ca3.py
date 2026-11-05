import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    rank = 38
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 38
    V = [[0] * rank for _ in range(m * p)] # 20 x 38
    W = [[0] * rank for _ in range(p * n)] # 10 x 38
    
    col = 0
    
    # Helper to add a rank-1 term
    # a_indices: list of (i, j, coeff) for A
    # b_indices: list of (j, k, coeff) for B
    # c_indices: list of (i, k, coeff) for C
    def add_term(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        if col >= rank:
            return
        for i, j, c in a_coeffs:
            U[i * m + j][col] += c
        for j, k, c in b_coeffs:
            V[j * p + k][col] += c
        for i, k, c in c_coeffs:
            W[k * n + i][col] += c
        col += 1

    # Strassen for 2x2x2
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # M2 = (A10+A11)B00 -> C10 += 1
    # M3 = A00(B01-B11) -> C01 += 1
    # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    # M7 = (A01-A11)(B10+B11) -> C00 += 1
    
    def add_strassen_block(a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset):
        # a_i_offset: 0 or 1 (row of A)
        # a_j_offset: 0 or 2 (col of A, 0 for first 2, 2 for next 2)
        # b_j_offset: 0 or 2 (row of B)
        # b_k_offset: 0 or 2 (col of B, 0 for first 2, 2 for next 3)
        # c_i_offset: 0 or 1 (row of C)
        # c_k_offset: 0 or 2 (col of C)
        
        # M1
        add_term(
            [(a_i_offset, a_j_offset, 1), (a_i_offset+1, a_j_offset+1, 1)],
            [(b_j_offset, b_k_offset, 1), (b_j_offset+1, b_k_offset+1, 1)],
            [(c_i_offset, c_k_offset, 1), (c_i_offset+1, c_k_offset+1, 1)]
        )
        # M2
        add_term(
            [(a_i_offset+1, a_j_offset, 1), (a_i_offset+1, a_j_offset+1, 1)],
            [(b_j_offset, b_k_offset, 1)],
            [(c_i_offset+1, c_k_offset, 1)]
        )
        # M3
        add_term(
            [(a_i_offset, a_j_offset, 1)],
            [(b_j_offset, b_k_offset+1, 1), (b_j_offset+1, b_k_offset+1, -1)],
            [(c_i_offset, c_k_offset+1, 1)]
        )
        # M4
        add_term(
            [(a_i_offset+1, a_j_offset+1, 1)],
            [(b_j_offset+1, b_k_offset, 1), (b_j_offset, b_k_offset, -1)],
            [(c_i_offset, c_k_offset, 1), (c_i_offset+1, c_k_offset, 1)]
        )
        # M5
        add_term(
            [(a_i_offset, a_j_offset, 1), (a_i_offset, a_j_offset+1, 1)],
            [(b_j_offset+1, b_k_offset+1, 1)],
            [(c_i_offset, c_k_offset, -1), (c_i_offset, c_k_offset+1, 1)]
        )
        # M6
        add_term(
            [(a_i_offset+1, a_j_offset, 1), (a_i_offset, a_j_offset, -1)],
            [(b_j_offset, b_k_offset, 1), (b_j_offset, b_k_offset+1, 1)],
            [(c_i_offset+1, c_k_offset+1, 1)]
        )
        # M7
        add_term(
            [(a_i_offset, a_j_offset+1, 1), (a_i_offset+1, a_j_offset+1, -1)],
            [(b_j_offset+1, b_k_offset, 1), (b_j_offset+1, b_k_offset+1, 1)],
            [(c_i_offset, c_k_offset, 1)]
        )

    # Schoolbook for 2x2x3
    def add_schoolbook_block(a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset):
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    add_term(
                        [(a_i_offset + i, a_j_offset + j, 1)],
                        [(b_j_offset + j, b_k_offset + k, 1)],
                        [(c_i_offset + i, c_k_offset + k, 1)]
                    )

    # Block 1: A1 (rows 0-1, cols 0-1) * B11 (rows 0-1, cols 0-1) -> C11 (rows 0-1, cols 0-1)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A2 (rows 0-1, cols 2-3) * B21 (rows 2-3, cols 0-1) -> C21 (rows 0-1, cols 0-1)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 3: A1 (rows 0-1, cols 0-1) * B12 (rows 0-1, cols 2-4) -> C12 (rows 0-1, cols 2-4)
    add_schoolbook_block(0, 0, 0, 2, 0, 2)
    
    # Block 4: A2 (rows 0-1, cols 2-3) * B22 (rows 2-3, cols 2-4) -> C22 (rows 0-1, cols 2-4)
    add_schoolbook_block(0, 2, 2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
