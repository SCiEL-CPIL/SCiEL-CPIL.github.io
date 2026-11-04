import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x rank
    V = [[0] * rank for _ in range(m * p)]  # 32 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank
    
    col = 0
    
    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        """Add one rank-1 term."""
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Adds Strassen decomposition for a 2x2 A block times 2x2 B block,
        contributing to a 2x2 C block.
        """
        u_a = a_rows[0] * m + a_cols[0]
        u_b = a_rows[0] * m + a_cols[1]
        u_c = a_rows[1] * m + a_cols[0]
        u_d = a_rows[1] * m + a_cols[1]
        
        v_e = b_rows[0] * p + b_cols[0]
        v_f = b_rows[0] * p + b_cols[1]
        v_g = b_rows[1] * p + b_cols[0]
        v_h = b_rows[1] * p + b_cols[1]
        
        w_c11 = c_cols[0] * n + c_rows[0]
        w_c12 = c_cols[1] * n + c_rows[0]
        w_c21 = c_cols[0] * n + c_rows[1]
        w_c22 = c_cols[1] * n + c_rows[1]
        
        # M1 = (a+d)(e+h) -> C11 += M1, C22 += M1
        add_rank_unit(
            {u_a: 1, u_d: 1},
            {v_e: 1, v_h: 1},
            {w_c11: 1, w_c22: 1}
        )
        # M2 = (c+d)e -> C21 += M2, C22 -= M2
        add_rank_unit(
            {u_c: 1, u_d: 1},
            {v_e: 1},
            {w_c21: 1, w_c22: -1}
        )
        # M3 = a(f-h) -> C12 += M3, C22 += M3
        add_rank_unit(
            {u_a: 1},
            {v_f: 1, v_h: -1},
            {w_c12: 1, w_c22: 1}
        )
        # M4 = d(g-e) -> C11 += M4, C21 += M4
        add_rank_unit(
            {u_d: 1},
            {v_g: 1, v_e: -1},
            {w_c11: 1, w_c21: 1}
        )
        # M5 = (a+b)h -> C11 -= M5, C12 += M5
        add_rank_unit(
            {u_a: 1, u_b: 1},
            {v_h: 1},
            {w_c11: -1, w_c12: 1}
        )
        # M6 = (c-a)(e+f) -> C22 += M6
        add_rank_unit(
            {u_c: 1, u_a: -1},
            {v_e: 1, v_f: 1},
            {w_c22: 1}
        )
        # M7 = (b-d)(g+h) -> C11 += M7
        add_rank_unit(
            {u_b: 1, u_d: -1},
            {v_g: 1, v_h: 1},
            {w_c11: 1}
        )
    
    # Part 1: A rows 0-1 (2x4) times B (4x8)
    a_cols_1 = [0, 1]
    a_cols_2 = [2, 3]
    b_rows_1 = [0, 1]
    b_rows_2 = [2, 3]
    
    for a_cols in [a_cols_1, a_cols_2]:
        for b_rows in [b_rows_1, b_rows_2]:
            for k_start in range(0, 8, 2):
                b_cols = [k_start, k_start + 1]
                c_rows = [0, 1]
                c_cols = [k_start, k_start + 1]
                add_strassen_block([0, 1], a_cols, b_rows, b_cols, c_rows, c_cols)
    
    # Part 2: A row 2 (1x4) times B (4x8) - schoolbook
    for k in range(8):
        for j in range(4):
            add_rank_unit(
                {2 * m + j: 1},
                {j * p + k: 1},
                {k * n + 2: 1}
            )
    
    if col != rank:
        raise Exception(f"Rank mismatch: used {col}, expected {rank}")
    
    return {"U": U, "V": V, "W": W}
