"""Matrix multiplication 3x4x6 - rank 66 decomposition via 3x2x2 blocks."""

def construct():
    n, m, p = 3, 4, 6
    
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for a_split in range(2):
        for b_split in range(3):
            a_cols = [2 * a_split, 2 * a_split + 1]
            b_rows = [2 * a_split, 2 * a_split + 1]
            b_cols = [2 * b_split, 2 * b_split + 1]
            
            a00 = 0 * m + a_cols[0]
            a01 = 0 * m + a_cols[1]
            a10 = 1 * m + a_cols[0]
            a11 = 1 * m + a_cols[1]
            a20 = 2 * m + a_cols[0]
            a21 = 2 * m + a_cols[1]
            
            b00 = b_rows[0] * p + b_cols[0]
            b01 = b_rows[0] * p + b_cols[1]
            b10 = b_rows[1] * p + b_cols[0]
            b11 = b_rows[1] * p + b_cols[1]
            
            c00 = b_cols[0] * n + 0
            c01 = b_cols[1] * n + 0
            c10 = b_cols[0] * n + 1
            c11 = b_cols[1] * n + 1
            c20 = b_cols[0] * n + 2
            c21 = b_cols[1] * n + 2
            
            def add_term(u_coeffs, v_coeffs, w_coeffs):
                nonlocal col
                for idx, coeff in u_coeffs:
                    U[idx][col] += coeff
                for idx, coeff in v_coeffs:
                    V[idx][col] += coeff
                for idx, coeff in w_coeffs:
                    W[idx][col] += coeff
                col += 1
            
            # 7 Strassen-like terms for 2x2x2 (top 2 rows)
            add_term([(a00,1),(a11,1)], [(b00,1),(b11,1)], [(c00,1),(c11,1)])
            add_term([(a10,1),(a11,1)], [(b00,1)], [(c10,1),(c11,-1)])
            add_term([(a00,1)], [(b01,1),(b11,-1)], [(c01,1),(c11,1)])
            add_term([(a11,1)], [(b10,1),(b00,-1)], [(c00,1),(c10,1)])
            add_term([(a00,1),(a01,1)], [(b11,1)], [(c00,-1),(c01,1)])
            add_term([(a10,1),(a00,-1)], [(b00,1),(b01,1)], [(c11,1)])
            add_term([(a01,1),(a11,-1)], [(b10,1),(b11,1)], [(c00,1)])
            
            # 4 schoolbook terms for bottom row
            add_term([(a20,1)], [(b00,1)], [(c20,1)])
            add_term([(a21,1)], [(b10,1)], [(c20,1)])
            add_term([(a20,1)], [(b01,1)], [(c21,1)])
            add_term([(a21,1)], [(b11,1)], [(c21,1)])
    
    return {"U": U, "V": V, "W": W}
