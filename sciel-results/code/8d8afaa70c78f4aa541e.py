import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 17 decomposition for 3x3x2:
    # A is 3x3, B is 3x2, C is 3x2
    # Split A into: A_top (2x2), A_col (2x1), A_row (1x2), A_scalar (1x1)
    # Split B into: B_top (2x2), B_row (1x2)
    # C_top (2x2) = A_top @ B_top + A_col @ B_row
    # C_bot (1x2) = A_row @ B_top + A_scalar @ B_row
    # 
    # A_top @ B_top: rank 7 (Strassen)
    # A_col @ B_row: rank 4 (schoolbook, 2*1*2)
    # A_row @ B_top: rank 4 (schoolbook, 1*2*2)
    # A_scalar @ B_row: rank 2 (schoolbook, 1*1*2)
    # Total: 17
    
    # Rank 9 for 3x3x1: schoolbook
    
    rank = 3 * 17 + 9  # = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1
    
    def add_strassen_2x2x2(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        """Add Strassen 2x2x2 decomposition.
        A indices: (a_i0+i, a_j0+j) for i,j in 0,1
        B indices: (b_j0+j, b_k0+k) for j,k in 0,1
        C indices: (c_i0+i, c_k0+k) -> W index (c_k0+k)*n + (c_i0+i)
        """
        def widx(ci, ck):
            return (c_k0 + ck) * n + (c_i0 + ci)
        
        def uidx(ai, aj):
            return (a_i0 + ai) * m + (a_j0 + aj)
        
        def vidx(bj, bk):
            return (b_j0 + bj) * p + (b_k0 + bk)
        
        def add_rank1(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs:
                U[idx][col] = val
            for idx, val in v_coeffs:
                V[idx][col] = val
            for idx, val in w_coeffs:
                W[idx][col] = val
            col += 1
        
        # M1: u = {A11:1, A22:1}, v = {B11:1, B22:1}, w = {C11:1, C22:1}
        add_rank1(
            [(uidx(0,0), 1), (uidx(1,1), 1)],
            [(vidx(0,0), 1), (vidx(1,1), 1)],
            [(widx(0,0), 1), (widx(1,1), 1)]
        )
        
        # M2: u = {A21:1, A22:1}, v = {B11:1}, w = {C21:1}
        add_rank1(
            [(uidx(1,0), 1), (uidx(1,1), 1)],
            [(vidx(0,0), 1)],
            [(widx(1,0), 1)]
        )
        
        # M3: u = {A11:1}, v = {B12:1, B22:-1}, w = {C12:1}
        add_rank1(
            [(uidx(0,0), 1)],
            [(vidx(0,1), 1), (vidx(1,1), -1)],
            [(widx(0,1), 1)]
        )
        
        # M4: u = {A22:1}, v = {B21:1, B11:-1}, w = {C11:1, C21:1}
        add_rank1(
            [(uidx(1,1), 1)],
            [(vidx(1,0), 1), (vidx(0,0), -1)],
            [(widx(0,0), 1), (widx(1,0), 1)]
        )
        
        # M5: u = {A11:1, A12:1}, v = {B22:1}, w = {C11:-1, C12:1}
        add_rank1(
            [(uidx(0,0), 1), (uidx(0,1), 1)],
            [(vidx(1,1), 1)],
            [(widx(0,0), -1), (widx(0,1), 1)]
        )
        
        # M6: u = {A21:1, A11:-1}, v = {B11:1, B12:1}, w = {C22:1}
        add_rank1(
            [(uidx(1,0), 1), (uidx(0,0), -1)],
            [(vidx(0,0), 1), (vidx(0,1), 1)],
            [(widx(1,1), 1)]
        )
        
        # M7: u = {A12:1, A22:-1}, v = {B21:1, B22:1}, w = {C11:1}
        add_rank1(
            [(uidx(0,1), 1), (uidx(1,1), -1)],
            [(vidx(1,0), 1), (vidx(1,1), 1)],
            [(widx(0,0), 1)]
        )
    
    def add_schoolbook_2x1x2(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        for i in range(2):
            for k in range(2):
                add_term(
                    (a_i0 + i) * m + a_j0,
                    b_j0 * p + (b_k0 + k),
                    (c_k0 + k) * n + (c_i0 + i)
                )
    
    def add_schoolbook_1x2x2(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        for j in range(2):
            for k in range(2):
                add_term(
                    a_i0 * m + (a_j0 + j),
                    (b_j0 + j) * p + (b_k0 + k),
                    (c_k0 + k) * n + c_i0
                )
    
    def add_schoolbook_1x1x2(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        for k in range(2):
            add_term(
                a_i0 * m + a_j0,
                b_j0 * p + (b_k0 + k),
                (c_k0 + k) * n + c_i0
            )
    
    def add_3x3x2_block(b_col_offset):
        add_strassen_2x2x2(0, 0, 0, b_col_offset, 0, b_col_offset)
        add_schoolbook_2x1x2(0, 2, 2, b_col_offset, 0, b_col_offset)
        add_schoolbook_1x2x2(2, 0, 0, b_col_offset, 2, b_col_offset)
        add_schoolbook_1x1x2(2, 2, 2, b_col_offset, 2, b_col_offset)
    
    for block in range(3):
        add_3x3x2_block(block * 2)
    
    b_col_offset = 6
    for i in range(3):
        for j in range(3):
            add_term(
                i * m + j,
                j * p + b_col_offset,
                b_col_offset * n + i
            )
    
    return {"U": U, "V": V, "W": W}
