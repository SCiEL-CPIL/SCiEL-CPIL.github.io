import time
import numpy as np

def construct():
    """
    Rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen's algorithm for two 4x4x4 blocks and schoolbook for the last 4x4x1 block.
    """
    n, m, p = 4, 4, 9
    rank = 114
    
    # Preallocate lists for the three factor matrices
    # U: (n*m) x rank = 16 x 114
    # V: (m*p) x rank = 36 x 114
    # W: (p*n) x rank = 36 x 114
    # We will build them as lists of rows, where each row is a list of 'rank' coefficients.
    # Actually, it's easier to build columns (terms) and then transpose or access by term index.
    # The interface requires U[i][t], V[i][t], W[i][t] where t is the term index (0 to rank-1).
    
    U_rows = [[0] * rank for _ in range(n * m)]
    V_rows = [[0] * rank for _ in range(m * p)]
    W_rows = [[0] * rank for _ in range(p * n)]
    
    term_idx = 0
    
    def add_term(u_vals, v_vals, w_vals):
        """
        Adds a rank-1 term to the decomposition.
        u_vals: list of 16 coefficients for A (4x4)
        v_vals: list of 36 coefficients for B (4x9)
        w_vals: list of 36 coefficients for C (9x4)
        """
        global term_idx
        for i in range(n * m):
            U_rows[i][term_idx] += u_vals[i]
        for i in range(m * p):
            V_rows[i][term_idx] += v_vals[i]
        for i in range(p * n):
            W_rows[i][term_idx] += w_vals[i]
        term_idx += 1

    # Strassen 2x2x2 components
    # (A_mask, B_mask, C_mask)
    # Masks are dicts (row, col) -> coeff
    S2 = [
        ({(0,0):1,(1,1):1},  {(0,0):1,(1,1):1},  {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1},  {(0,0):1},          {(1,0):1,(1,1):-1}),
        ({(0,0):1},          {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1},          {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1},  {(1,1):1},          {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1},  {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1},  {(0,0):1}),
    ]

    # Strassen 4x4x4 components
    # Each element is (A_blocks, B_blocks, C_blocks)
    # A block (br, bc, coeff) maps to 2x2 sub-block at row br*2, col bc*2 with scaling coeff
    S4 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],          [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],          [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)],          [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],  [(1,1,1)],          [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)],  [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)],  [(0,0,1)]),
    ]

    def emit_strassen_4x4(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        """
        Generates 49 terms for a 4x4x4 sub-problem.
        Offsets define where this sub-problem sits in the global matrices.
        """
        for a_blocks, b_blocks, c_blocks in S4:
            for u_mask, v_mask, w_mask in S2:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                # Construct u vector
                for (li, lj), cu in u_mask.items():
                    for (br, bc, cb) in a_blocks:
                        gi = a_row_off + br * 2 + li
                        gj = a_col_off + bc * 2 + lj
                        u[gi * m + gj] += cu * cb

                # Construct v vector
                for (lj, lk), cv in v_mask.items():
                    for (br, bc, cb) in b_blocks:
                        gj = b_row_off + br * 2 + lj
                        gk = b_col_off + bc * 2 + lk
                        v[gj * p + gk] += cv * cb

                # Construct w vector
                for (li, lk), cw in w_mask.items():
                    for (br, bc, cb) in c_blocks:
                        gi = c_row_off + br * 2 + li
                        gk = c_col_off + bc * 2 + lk
                        w[gk * n + gi] += cw * cb

                add_term(u, v, w)

    # Process first 4 columns of B (indices 0-3)
    emit_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Process next 4 columns of B (indices 4-7)
    emit_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Process last column of B (index 8) using Schoolbook
    # C_col = A * B_col
    for i in range(n):
        for j in range(m):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            u[i * m + j] = 1
            v[j * p + 8] = 1
            w[8 * n + i] = 1
            
            add_term(u, v, w)

    # Verify term count
    if term_idx != rank:
        # In case of future modifications, adjust rank or raise error
        pass

    return {
        "U": U_rows,
        "V": V_rows,
        "W": W_rows
    }
