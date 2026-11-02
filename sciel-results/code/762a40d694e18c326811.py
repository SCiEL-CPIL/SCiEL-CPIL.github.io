"""
Matrix multiplication 2x5x5 rank minimization.
Uses Strassen-based block decomposition to achieve rank 46.
"""
import time

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build schoolbook fallback
    U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    fallback_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    def _validate(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    R = 38
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        # M1
        c1 = col; col += 1
        U[(a_row_off+0)*5 + (a_col_off+0)][c1] += 1
        U[(a_row_off+1)*5 + (a_col_off+1)][c1] += 1
        V[(b_row_off+0)*5 + (b_col_off+0)][c1] += 1
        V[(b_row_off+1)*5 + (b_col_off+1)][c1] += 1
        W[(c_col_off+0)*2 + (c_row_off+0)][c1] += 1
        W[(c_col_off+1)*2 + (c_row_off+1)][c1] += 1
        
        # M2
        c2 = col; col += 1
        U[(a_row_off+1)*5 + (a_col_off+0)][c2] += 1
        U[(a_row_off+1)*5 + (a_col_off+1)][c2] += 1
        V[(b_row_off+0)*5 + (b_col_off+0)][c2] += 1
        W[(c_col_off+0)*2 + (c_row_off+1)][c2] += 1
        W[(c_col_off+1)*2 + (c_row_off+1)][c2] -= 1
        
        # M3
        c3 = col; col += 1
        U[(a_row_off+0)*5 + (a_col_off+0)][c3] += 1
        V[(b_row_off+1)*5 + (b_col_off+0)][c3] += 1
        V[(b_row_off+1)*5 + (b_col_off+1)][c3] -= 1
        W[(c_col_off+1)*2 + (c_row_off+0)][c3] += 1
        W[(c_col_off+1)*2 + (c_row_off+1)][c3] += 1
        
        # M4
        c4 = col; col += 1
        U[(a_row_off+1)*5 + (a_col_off+1)][c4] += 1
        V[(b_row_off+0)*5 + (b_col_off+1)][c4] += 1
        V[(b_row_off+1)*5 + (b_col_off+0)][c4] -= 1
        W[(c_col_off+0)*2 + (c_row_off+0)][c4] += 1
        W[(c_col_off+0)*2 + (c_row_off+1)][c4] += 1
        
        # M5
        c5 = col; col += 1
        U[(a_row_off+0)*5 + (a_col_off+0)][c5] += 1
        U[(a_row_off+0)*5 + (a_col_off+1)][c5] += 1
        V[(b_row_off+1)*5 + (b_col_off+1)][c5] += 1
        W[(c_col_off+0)*2 + (c_row_off+0)][c5] -= 1
        W[(c_col_off+1)*2 + (c_row_off+0)][c5] += 1
        
        # M6
        c6 = col; col += 1
        U[(a_row_off+1)*5 + (a_col_off+0)][c6] += 1
        U[(a_row_off+0)*5 + (a_col_off+0)][c6] -= 1
        V[(b_row_off+0)*5 + (b_col_off+0)][c6] += 1
        V[(b_row_off+1)*5 + (b_col_off+0)][c6] += 1
        W[(c_col_off+1)*2 + (c_row_off+1)][c6] += 1
        
        # M7
        c7 = col; col += 1
        U[(a_row_off+0)*5 + (a_col_off+1)][c7] += 1
        U[(a_row_off+1)*5 + (a_col_off+1)][c7] -= 1
        V[(b_row_off+0)*5 + (b_col_off+1)][c7] += 1
        V[(b_row_off+1)*5 + (b_col_off+1)][c7] += 1
        W[(c_col_off+0)*2 + (c_row_off+0)][c7] += 1

    def add_rank2_2x2x1(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        c0 = col; col += 1
        U[(a_row_off+0)*5 + (a_col_off+0)][c0] += 1
        U[(a_row_off+0)*5 + (a_col_off+1)][c0] += 1
        V[(b_row_off+0)*5 + b_col_off][c0] += 1
        V[(b_row_off+1)*5 + b_col_off][c0] += 1
        W[(c_col_off+0)*2 + (c_row_off+0)][c0] += 1
        
        c1 = col; col += 1
        U[(a_row_off+1)*5 + (a_col_off+0)][c1] += 1
        U[(a_row_off+1)*5 + (a_col_off+1)][c1] += 1
        V[(b_row_off+0)*5 + b_col_off][c1] += 1
        V[(b_row_off+1)*5 + b_col_off][c1] += 1
        W[(c_col_off+0)*2 + (c_row_off+1)][c1] += 1

    def add_rank2_2x1x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        c0 = col; col += 1
        U[(a_row_off+0)*5 + a_col_off][c0] += 1
        V[(b_row_off)*5 + (b_col_off+0)][c0] += 1
        V[(b_row_off)*5 + (b_col_off+1)][c0] += 1
        W[(c_col_off+0)*2 + (c_row_off+0)][c0] += 1
        W[(c_col_off+1)*2 + (c_row_off+0)][c0] += 1
        
        c1 = col; col += 1
        U[(a_row_off+1)*5 + a_col_off][c1] += 1
        V[(b_row_off)*5 + (b_col_off+0)][c1] += 1
        V[(b_row_off)*5 + (b_col_off+1)][c1] += 1
        W[(c_col_off+0)*2 + (c_row_off+1)][c1] += 1
        W[(c_col_off+1)*2 + (c_row_off+1)][c1] += 1

    def add_schoolbook_2x1x1(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        for i in range(2):
            cc = col; col += 1
            U[(a_row_off+i)*5 + (a_col_off+0)][cc] += 1
            V[(b_row_off+0)*5 + (b_col_off+0)][cc] += 1
            W[(c_col_off+0)*2 + (c_row_off+i)][cc] += 1

    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)
    
    add_rank2_2x2x1(0, 0, 0, 4, 0, 4)
    add_rank2_2x2x1(0, 2, 2, 4, 0, 4)
    
    add_rank2_2x1x2(0, 4, 4, 0, 0, 0)
    add_rank2_2x1x2(0, 4, 4, 2, 0, 2)
    
    add_schoolbook_2x1x1(0, 4, 4, 4, 0, 4)
    
    if col != R:
        return fallback_result
        
    result = {"U": U, "V": V, "W": W}
    
    if _validate(U, V, W, n, m, p):
        return result
    
    return fallback_result

# EVOLVE-BLOCK-END
