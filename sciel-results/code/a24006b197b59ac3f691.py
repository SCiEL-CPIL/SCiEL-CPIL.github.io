"""Matrix multiplication 3x4x6 - rank 66 decomposition via 3x2x2 blocks."""

def construct():
    n, m, p = 3, 4, 6
    
    # Total rank = 2 (A splits) * 3 (B splits) * 11 (rank of 3x2x2) = 66
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Loop over A splits (0, 1) and B splits (0, 1, 2)
    for a_split in range(2):
        for b_split in range(3):
            # Determine global indices for this 3x2x2 block
            # A: rows 0,1,2; cols 2*a_split, 2*a_split+1
            # B: rows 2*a_split, 2*a_split+1; cols 2*b_split, 2*b_split+1
            # C: rows 0,1,2; cols 2*b_split, 2*b_split+1
            
            a_cols = [2 * a_split, 2 * a_split + 1]
            b_rows = [2 * a_split, 2 * a_split + 1]
            b_cols = [2 * b_split, 2 * b_split + 1]
            
            # Helper to add a term
            def add_term(u_coeffs, v_coeffs, w_coeffs):
                nonlocal col
                # u_coeffs: list of (global_a_idx, coeff)
                # v_coeffs: list of (global_b_idx, coeff)
                # w_coeffs: list of (global_c_idx, coeff)
                for idx, coeff in u_coeffs:
                    U[idx][col] += coeff
                for idx, coeff in v_coeffs:
                    V[idx][col] += coeff
                for idx, coeff in w_coeffs:
                    W[idx][col] += coeff
                col += 1
            
            # Strassen terms for top 2x2x2
            # A_top indices: (0, a_cols[0]), (0, a_cols[1]), (1, a_cols[0]), (1, a_cols[1])
            # B indices: (b_rows[0], b_cols[0]), (b_rows[0], b_cols[1]), (b_rows[1], b_cols[0]), (b_rows[1], b_cols[1])
            
            a00 = 0 * m + a_cols[0]
            a01 = 0 * m + a_cols[1]
            a10 = 1 * m + a_cols[0]
            a11 = 1 * m + a_cols[1]
            
            b00 = b_rows[0] * p + b_cols[0]
            b01 = b_rows[0] * p + b_cols[1]
            b10 = b_rows[1] * p + b_cols[0]
            b11 = b_rows[1] * p + b_cols[1]
            
            # C indices for output of this block
            # C[0, b_cols[0]], C[0, b_cols[1]], C[1, b_cols[0]], C[1, b_cols[1]], C[2, b_cols[0]], C[2, b_cols[1]]
            c00 = b_cols[0] * n + 0
            c01 = b_cols[1] * n + 0
            c10 = b_cols[0] * n + 1
            c11 = b_cols[1] * n + 1
            c20 = b_cols[0] * n + 2
            c21 = b_cols[1] * n + 2
            
            # M1 = (a00 + a11)(b00 + b11)
            add_term(
                [(a00, 1), (a11, 1)],
                [(b00, 1), (b11, 1)],
                [(c00, 1), (c11, 1)]
            )
            
            # M2 = (a10 + a11)b00
            add_term(
                [(a10, 1), (a11, 1)],
                [(b00, 1)],
                [(c10, 1), (c11, -1)]
            )
            
            # M3 = a00(b01 - b11)
            add_term(
                [(a00, 1)],
                [(b01, 1), (b11, -1)],
                [(c01, 1), (c11, 1)]
            )
            
            # M4 = a11(b10 - b00)
            add_term(
                [(a11, 1)],
                [(b10, 1), (b00, -1)],
                [(c00, 1), (c10, 1)]
            )
            
            # M5 = (a00 + a01)b11
            add_term(
                [(a00, 1), (a01, 1)],
                [(b11, 1)],
                [(c00, -1), (c01, 1)]
            )
            
            # M6 = (a10 - a00)(b00 + b01)
            add_term(
                [(a10, 1), (a00, -1)],
                [(b00, 1), (b01, 1)],
                [(c11, 1)]
            )
            
            # M7 = (a01 - a11)(b10 + b11)
            add_term(
                [(a01, 1), (a11, -1)],
                [(b10, 1), (b11, 1)],
                [(c00, 1)]
            )
            
            # Schoolbook terms for bottom row
            a20 = 2 * m + a_cols[0]
            a21 = 2 * m + a_cols[1]
            
            # M8 = a20 b00
            add_term(
                [(a20, 1)],
                [(b00, 1)],
                [(c20, 1)]
            )
            
            # M9 = a21 b10
            add_term(
                [(a21, 1)],
                [(b10, 1)],
                [(c20, 1)]
            )
            
            # M10 = a20 b01
            add_term(
                [(a20, 1)],
                [(b01, 1)],
                [(c21, 1)]
            )
            
            # M11 = a21 b11
            add_term(
                [(a21, 1)],
                [(b11, 1)],
                [(c21, 1)]
            )

    return {"U": U, "V": V, "W": W}
