"""Matrix multiplication 2x4x7 tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # 6 blocks of 2x2x2 Strassen (7 each) = 42
    # 2 blocks of 2x2x1 (4 each) = 8
    total_rank = 6 * 7 + 2 * 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col_offset
        # M1: (a+d)(e+h)
        u_idx = [a_row_start*m + a_col_start, (a_row_start+1)*m + (a_col_start+1)]
        v_idx = [b_row_start*p + b_col_start, (b_row_start+1)*p + (b_col_start+1)]
        w_idx = [(c_col_start)*n + c_row_start, (c_col_start+1)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] += 1
        W[w_idx[0]][col_offset] += 1
        W[w_idx[1]][col_offset] += 1
        col_offset += 1
        
        # M2: (c+d)e
        u_idx = [(a_row_start+1)*m + a_col_start, (a_row_start+1)*m + (a_col_start+1)]
        v_idx = [b_row_start*p + b_col_start]
        w_idx = [(c_col_start)*n + (c_row_start+1), (c_col_start+1)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        W[w_idx[0]][col_offset] += 1
        W[w_idx[1]][col_offset] -= 1
        col_offset += 1
        
        # M3: a(f-h)
        u_idx = [a_row_start*m + a_col_start]
        v_idx = [b_row_start*p + (b_col_start+1), (b_row_start+1)*p + (b_col_start+1)]
        w_idx = [(c_col_start+1)*n + c_row_start, (c_col_start+1)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] -= 1
        W[w_idx[0]][col_offset] += 1
        W[w_idx[1]][col_offset] += 1
        col_offset += 1
        
        # M4: d(g-e)
        u_idx = [(a_row_start+1)*m + (a_col_start+1)]
        v_idx = [(b_row_start+1)*p + b_col_start, b_row_start*p + b_col_start]
        w_idx = [(c_col_start)*n + c_row_start, (c_col_start)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] -= 1
        W[w_idx[0]][col_offset] += 1
        W[w_idx[1]][col_offset] += 1
        col_offset += 1
        
        # M5: (a+b)h
        u_idx = [a_row_start*m + a_col_start, a_row_start*m + (a_col_start+1)]
        v_idx = [(b_row_start+1)*p + (b_col_start+1)]
        w_idx = [(c_col_start)*n + c_row_start, (c_col_start+1)*n + c_row_start]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        W[w_idx[0]][col_offset] -= 1
        W[w_idx[1]][col_offset] += 1
        col_offset += 1
        
        # M6: (c-a)(e+f)
        u_idx = [(a_row_start+1)*m + a_col_start, a_row_start*m + a_col_start]
        v_idx = [b_row_start*p + b_col_start, b_row_start*p + (b_col_start+1)]
        w_idx = [(c_col_start+1)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] -= 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] += 1
        W[w_idx[0]][col_offset] += 1
        col_offset += 1
        
        # M7: (b-d)(g+h)
        u_idx = [a_row_start*m + (a_col_start+1), (a_row_start+1)*m + (a_col_start+1)]
        v_idx = [(b_row_start+1)*p + b_col_start, (b_row_start+1)*p + (b_col_start+1)]
        w_idx = [(c_col_start)*n + c_row_start]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] -= 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] += 1
        W[w_idx[0]][col_offset] += 1
        col_offset += 1

    def add_dot(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col_offset
        # C_0 = A_00 B_00 + A_01 B_10
        # C_1 = A_10 B_00 + A_11 B_10
        
        # Term 1: A_00 B_00 -> C_0
        U[a_row_start*m + a_col_start][col_offset] += 1
        V[b_row_start*p + b_col_start][col_offset] += 1
        W[c_col_start*n + c_row_start][col_offset] += 1
        col_offset += 1
        
        # Term 2: A_01 B_10 -> C_0
        U[a_row_start*m + (a_col_start+1)][col_offset] += 1
        V[(b_row_start+1)*p + b_col_start][col_offset] += 1
        W[c_col_start*n + c_row_start][col_offset] += 1
        col_offset += 1
        
        # Term 3: A_10 B_00 -> C_1
        U[(a_row_start+1)*m + a_col_start][col_offset] += 1
        V[b_row_start*p + b_col_start][col_offset] += 1
        W[c_col_start*n + (c_row_start+1)][col_offset] += 1
        col_offset += 1
        
        # Term 4: A_11 B_10 -> C_1
        U[(a_row_start+1)*m + (a_col_start+1)][col_offset] += 1
        V[(b_row_start+1)*p + b_col_start][col_offset] += 1
        W[c_col_start*n + (c_row_start+1)][col_offset] += 1
        col_offset += 1

    # 6 Strassen blocks for the 2x4x6 part
    add_strassen(0, 0, 0, 0, 0, 0)   # rows 0-1, cols 0-1 of A; rows 0-1, cols 0-1 of B -> rows 0-1, cols 0-1 of C
    add_strassen(0, 0, 0, 2, 0, 2)   # rows 0-1, cols 0-1 of A; rows 0-1, cols 2-3 of B -> rows 0-1, cols 2-3 of C
    add_strassen(0, 2, 2, 0, 0, 0)   # rows 0-1, cols 2-3 of A; rows 2-3, cols 0-1 of B -> rows 0-1, cols 0-1 of C
    add_strassen(0, 2, 2, 2, 0, 2)   # rows 0-1, cols 2-3 of A; rows 2-3, cols 2-3 of B -> rows 0-1, cols 2-3 of C
    add_strassen(0, 0, 0, 4, 0, 4)   # rows 0-1, cols 0-1 of A; rows 0-1, cols 4-5 of B -> rows 0-1, cols 4-5 of C
    add_strassen(0, 2, 2, 4, 0, 4)   # rows 0-1, cols 2-3 of A; rows 2-3, cols 4-5 of B -> rows 0-1, cols 4-5 of C
    
    # 2 dot blocks for the 2x4x1 part (col 6 of B)
    add_dot(0, 0, 0, 6, 0, 6)   # rows 0-1, cols 0-1 of A; rows 0-1, col 6 of B -> rows 0-1, col 6 of C
    add_dot(0, 2, 2, 6, 0, 6)   # rows 0-1, cols 2-3 of A; rows 2-3, col 6 of B -> rows 0-1, col 6 of C

    return {"U": U, "V": V, "W": W}
