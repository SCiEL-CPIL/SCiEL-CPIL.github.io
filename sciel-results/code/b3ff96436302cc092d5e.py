"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen for 2x2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Two 2x2x6 blocks
    for block_id in range(2):
        a_off = block_id * 2
        b_off = block_id * 2
        # Three 2x2x2 sub-blocks for the 6 columns
        for sub_block in range(3):
            b_col_off = sub_block * 2
            c_col_off = sub_block * 2
            
            # Helper to add a rank-1 term
            def add_term(u_idxs, u_coeffs, v_idxs, v_coeffs, w_idxs, w_coeffs):
                nonlocal col
                for idx, coeff in zip(u_idxs, u_coeffs):
                    U[idx][col] += coeff
                for idx, coeff in zip(v_idxs, v_coeffs):
                    V[idx][col] += coeff
                for idx, coeff in zip(w_idxs, w_coeffs):
                    W[idx][col] += coeff
            
            # Term 1: M1 = (a00+a11)(b00+b11) -> c00+c11
            add_term(
                [0*m+a_off, 1*m+a_off+1], [1, 1],
                [b_off*p+b_col_off, (b_off+1)*p+b_col_off+1], [1, 1],
                [c_col_off*n+0, (c_col_off+1)*n+1], [1, 1]
            )
            
            # Term 2: M2 = (a10+a11)b00 -> c10+c11
            add_term(
                [1*m+a_off, 1*m+a_off+1], [1, 1],
                [b_off*p+b_col_off], [1],
                [c_col_off*n+1, (c_col_off+1)*n+1], [1, 1]
            )
            
            # Term 3: M3 = a00(b10-b11) -> c01+c11
            add_term(
                [0*m+a_off], [1],
                [(b_off+1)*p+b_col_off, (b_off+1)*p+b_col_off+1], [1, -1],
                [(c_col_off+1)*n+0, (c_col_off+1)*n+1], [1, 1]
            )
            
            # Term 4: M4 = a11(b01-b00) -> c00+c10
            add_term(
                [1*m+a_off+1], [1],
                [b_off*p+b_col_off+1, b_off*p+b_col_off], [1, -1],
                [c_col_off*n+0, c_col_off*n+1], [1, 1]
            )
            
            # Term 5: M5 = (a00+a01)b11 -> -c00+c01
            add_term(
                [0*m+a_off, 0*m+a_off+1], [1, 1],
                [(b_off+1)*p+b_col_off+1], [1],
                [c_col_off*n+0, (c_col_off+1)*n+0], [-1, 1]
            )
            
            # Term 6: M6 = (a10-a00)(b00+b01) -> c11
            add_term(
                [1*m+a_off, 0*m+a_off], [1, -1],
                [b_off*p+b_col_off, b_off*p+b_col_off+1], [1, 1],
                [(c_col_off+1)*n+1], [1]
            )
            
            # Term 7: M7 = (a01-a11)(b10+b11) -> c00
            add_term(
                [0*m+a_off+1, 1*m+a_off+1], [1, -1],
                [(b_off+1)*p+b_col_off, (b_off+1)*p+b_col_off+1], [1, 1],
                [c_col_off*n+0], [1]
            )
            
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
