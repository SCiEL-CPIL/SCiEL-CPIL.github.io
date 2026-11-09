import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will fill U, V, W with the decomposition.
    # We need a helper to add a term.
    # term: (u_indices, v_indices, w_indices, coeff)
    # u_indices: list of (i, j, val)
    # v_indices: list of (j, k, val)
    # w_indices: list of (i, k, val)
    
    col = 0
    
    def add_term(u, v, w):
        nonlocal col
        for i, j, val in u:
            U[i * m + j][col] += val
        for j, k, val in v:
            V[j * p + k][col] += val
        for i, k, val in w:
            W[k * n + i][col] += val
        col += 1

    # Strassen 2x2x2
    def strassen_2x2x2(a_offset, b_offset, c_offset, a_rows, b_cols, c_rows, c_cols):
        # a_rows: 2 indices for A rows
        # b_cols: 2 indices for B cols
        # c_rows: 2 indices for C rows
        # c_cols: 2 indices for C cols
        # b_rows: 2 indices for B rows (implicit from a_cols?)
        # a_cols: 2 indices for A cols (implicit from b_rows?)
        # We need to pass all indices.
        pass

    # This is getting complex to generalize.
    # I will hardcode the 44 terms based on the structure.
    
    # Structure:
    # 1. A1 (rows 0,1,2; cols 0,1) * B1 (rows 0,1; cols 0,1,2,3)
    #    1a. A1 * B1a (cols 0,1)
    #        1a1. A1_top (rows 0,1) * B1a (2x2x2, rank 7)
    #        1a2. A1_bot (row 2) * B1a (1x2x2, rank 4)
    #    1b. A1 * B1b (cols 2,3)
    #        1b1. A1_top (rows 0,1) * B1b (2x2x2, rank 7)
    #        1b2. A1_bot (row 2) * B1b (1x2x2, rank 4)
    # 2. A2 (rows 0,1,2; cols 2,3) * B2 (rows 2,3; cols 0,1,2,3)
    #    2a. A2 * B2a (cols 0,1)
    #        2a1. A2_top (rows 0,1) * B2a (2x2x2, rank 7)
    #        2a2. A2_bot (row 2) * B2a (1x2x2, rank 4)
    #    2b. A2 * B2b (cols 2,3)
    #        2b1. A2_top (rows 0,1) * B2b (2x2x2, rank 7)
    #        2b2. A2_bot (row 2) * B2b (1x2x2, rank 4)
    
    # Total 4 * (7 + 4) = 44.
    
    # Helper to add 2x2x2 Strassen
    def add_strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # a_rows: [r1, r2], a_cols: [c1, c2]
        # b_rows: [r1, r2], b_cols: [c1, c2]
        # c_rows: [r1, r2], c_cols: [c1, c2]
        
        # M1 = (A00 + A11)(B00 + B11)
        u = [(a_rows[0], a_cols[0], 1), (a_rows[1], a_cols[1], 1)]
        v = [(b_rows[0], b_cols[0], 1), (b_rows[1], b_cols[1], 1)]
        w = [(c_rows[0], c_cols[0], 1), (c_rows[1], c_cols[1], 1)] # Wait, M1 contributes to C00 and C11
        # C00 += M1, C11 += M1
        # We need to add 7 terms.
        
        # Term 1: M1
        add_term(
            [(a_rows[0], a_cols[0], 1), (a_rows[1], a_cols[1], 1)],
            [(b_rows[0], b_cols[0], 1), (b_rows[1], b_cols[1], 1)],
            [(c_rows[0], c_cols[0], 1), (c_rows[1], c_cols[1], 1)] # This is wrong. M1 is a scalar.
        )
        # The w vector for a scalar product contributing to C00 and C11?
        # No, the decomposition is $C_{ik} = \sum u_{ij} v_{jk} w_{ik}$.
        # If we have a scalar product $P = (\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk})$,
        # and it contributes to $C_{ik}$ with coefficient $w_{ik}$,
        # then the term is $u \otimes v \otimes w$.
        # For Strassen, $M_1$ contributes to $C_{00}$ with coeff 1 and $C_{11}$ with coeff 1.
        # So we need TWO terms?
        # No, $w$ can have multiple non-zero entries.
        # $w = [(c_rows[0], c_cols[0], 1), (c_rows[1], c_cols[1], 1)]$.
        # This works.
        
        # Let's implement the 7 terms for 2x2x2.
        
        # M1
        add_term(
            [(a_rows[0], a_cols[0], 1), (a_rows[1], a_cols[1], 1)],
            [(b_rows[0], b_cols[0], 1), (b_rows[1], b_cols[1], 1)],
            [(c_rows[0], c_cols[0], 1), (c_rows[1], c_cols[1], 1)]
        )
        # M2
        add_term(
            [(a_rows[1], a_cols[0], 1), (a_rows[1], a_cols[1], 1)],
            [(b_rows[0], b_cols[0], 1)],
            [(c_rows[0], c_cols[1], 1), (c_rows[1], c_cols[0], 1)]
        )
        # M3
        add_term(
            [(a_rows[0], a_cols[0], 1)],
            [(b_rows[1], b_cols[0], 1), (b_rows[1], b_cols[1], -1)],
            [(c_rows[0], c_cols[1], 1), (c_rows[1], c_cols[0], 1)]
        )
        # M4
        add_term(
            [(a_rows[1], a_cols[1], 1)],
            [(b_rows[0], b_cols[1], 1), (b_rows[0], b_cols[0], -1)],
            [(c_rows[0], c_cols[0], 1), (c_rows[1], c_cols[1], 1)]
        )
        # M5
        add_term(
            [(a_rows[0], a_cols[0], 1), (a_rows[0], a_cols[1], 1)],
            [(b_rows[1], b_cols[1], 1)],
            [(c_rows[1], c_cols[0], 1), (c_rows[0], c_cols[0], -1)]
        )
        # M6
        add_term(
            [(a_rows[1], a_cols[0], 1), (a_rows[0], a_cols[0], -1)],
            [(b_rows[0], b_cols[0], 1), (b_rows[0], b_cols[1], 1)],
            [(c_rows[1], c_cols[0], 1), (c_rows[1], c_cols[1], 1)]
        )
        # M7
        add_term(
            [(a_rows[0], a_cols[1], 1), (a_rows[1], a_cols[1], -1)],
            [(b_rows[1], b_cols[0], 1), (b_rows[1], b_cols[1], 1)],
            [(c_rows[0], c_cols[0], -1), (c_rows[1], c_cols[1], 1)]
        )

    # Helper to add 1x2x2 schoolbook
    def add_1x2x2(a_row, a_cols, b_rows, b_cols, c_row, c_cols):
        nonlocal col
        # C = A B
        # C0 = A0 B0 + A1 B1
        # C1 = A0 B2 + A1 B3
        # a_row: index
        # a_cols: [c1, c2]
        # b_rows: [r1, r2]
        # b_cols: [c1, c2]
        # c_row: index
        # c_cols: [c1, c2]
        
        add_term(
            [(a_row, a_cols[0], 1)],
            [(b_rows[0], b_cols[0], 1)],
            [(c_row, c_cols[0], 1)]
        )
        add_term(
            [(a_row, a_cols[1], 1)],
            [(b_rows[1], b_cols[0], 1)],
            [(c_row, c_cols[0], 1)]
        )
        add_term(
            [(a_row, a_cols[0], 1)],
            [(b_rows[0], b_cols[1], 1)],
            [(c_row, c_cols[1], 1)]
        )
        add_term(
            [(a_row, a_cols[1], 1)],
            [(b_rows[1], b_cols[1], 1)],
            [(c_row, c_cols[1], 1)]
        )

    # Now call them.
    
    # 1a1: A1_top (0,1) * B1a (0,1)
    add_strassen_2x2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # 1a2: A1_bot (2) * B1a (0,1)
    add_1x2x2(2, [0,1], [0,1], [0,1], 2, [0,1])
    
    # 1b1: A1_top (0,1) * B1b (0,1) but B cols are 2,3
    add_strassen_2x2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # 1b2: A1_bot (2) * B1b (0,1)
    add_1x2x2(2, [0,1], [0,1], [2,3], 2, [2,3])
    
    # 2a1: A2_top (0,1) * B2a (2,3)
    add_strassen_2x2x2([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    # 2a2: A2_bot (2) * B2a (2,3)
    add_1x2x2(2, [2,3], [2,3], [0,1], 2, [0,1])
    
    # 2b1: A2_top (0,1) * B2b (2,3)
    add_strassen_2x2x2([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    # 2b2: A2_bot (2) * B2b (2,3)
    add_1x2x2(2, [2,3], [2,3], [2,3], 2, [2,3])
    
    return {"U": U, "V": V, "W": W}

