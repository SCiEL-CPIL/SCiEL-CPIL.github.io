import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-66 exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    Uses Strassen's algorithm for 2x2x2 blocks and schoolbook for 1x2x2 cross terms.
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We'll build U, V, W incrementally
    U = []
    V = []
    W = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict: {(i,j): val}, v_dict: {(j,k): val}, w_dict: {(k*n+i): val}"""
        U.append(u_dict)
        V.append(v_dict)
        W.append(w_dict)
    
    def add_strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        """
        Strassen for 2x2 x 2x2 -> 2x2.
        A block: rows [a_row_off, a_row_off+2), cols [a_col_off, a_col_off+2)
        B block: rows [b_row_off, b_row_off+2), cols [b_col_off, b_col_off+2)
        C block: rows [c_row_off, c_row_off+2), cols [c_col_off, c_col_off+2)
        
        W index for C[i,j] is j*n + i (where i is row in 0..1, j is col in 0..1)
        So W index = (c_col_off + j)*n + (c_row_off + i)
        """
        # A indices: (row, col) in 2x2 block
        a00 = (a_row_off, a_col_off)
        a01 = (a_row_off, a_col_off + 1)
        a10 = (a_row_off + 1, a_col_off)
        a11 = (a_row_off + 1, a_col_off + 1)
        
        # B indices: (row, col) in 2x2 block
        b00 = (b_row_off, b_col_off)
        b01 = (b_row_off, b_col_off + 1)
        b10 = (b_row_off + 1, b_col_off)
        b11 = (b_row_off + 1, b_col_off + 1)
        
        # W indices: C[i,j] -> (c_col_off + j)*n + (c_row_off + i)
        w_c00 = (c_col_off + 0) * n + (c_row_off + 0)  # C[0,0]
        w_c01 = (c_col_off + 1) * n + (c_row_off + 0)  # C[0,1]
        w_c10 = (c_col_off + 0) * n + (c_row_off + 1)  # C[1,0]
        w_c11 = (c_col_off + 1) * n + (c_row_off + 1)  # C[1,1]
        
        # Strassen factors:
        # M1 = (A00+A11)(B00+B11)
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {w_c00: 1, w_c11: 1})
        
        # M2 = (A10+A11)B00
        add_term({a10: 1, a11: 1}, {b00: 1}, {w_c10: 1, w_c11: -1})
        
        # M3 = A00(B10-B11)
        add_term({a00: 1}, {b10: 1, b11: -1}, {w_c01: 1, w_c11: 1})
        
        # M4 = A11(B01-B10)
        add_term({a11: 1}, {b01: 1, b10: -1}, {w_c00: 1, w_c10: 1})
        
        # M5 = (A00+A01)B11
        add_term({a00: 1, a01: 1}, {b11: 1}, {w_c00: -1, w_c01: 1})
        
        # M6 = (A10-A00)(B00+B10)
        add_term({a10: 1, a00: -1}, {b00: 1, b10: 1}, {w_c11: 1})
        
        # M7 = (A01-A11)(B01+B11)
        add_term({a01: 1, a11: -1}, {b01: 1, b11: 1}, {w_c00: 1})
    
    def add_schoolbook_1x2x2(a_row, a_col_off, b_row_off, b_col_off, c_row, c_col_off):
        """
        Schoolbook for 1x2 x 2x2 -> 1x2.
        A: row a_row, cols [a_col_off, a_col_off+2)
        B: rows [b_row_off, b_row_off+2), cols [b_col_off, b_col_off+2)
        C: row c_row, cols [c_col_off, c_col_off+2)
        
        C[j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
        W index for C[i,j] = j*n + i
        """
        # Term: A[0,0] * B[0,0] -> C[0,0]
        add_term({(a_row, a_col_off): 1}, {(b_row_off, b_col_off): 1}, {(c_col_off + 0)*n + c_row: 1})
        
        # Term: A[0,0] * B[0,1] -> C[0,1]
        add_term({(a_row, a_col_off): 1}, {(b_row_off, b_col_off+1): 1}, {(c_col_off + 1)*n + c_row: 1})
        
        # Term: A[0,1] * B[1,0] -> C[0,0]
        add_term({(a_row, a_col_off+1): 1}, {(b_row_off+1, b_col_off): 1}, {(c_col_off + 0)*n + c_row: 1})
        
        # Term: A[0,1] * B[1,1] -> C[0,1]
        add_term({(a_row, a_col_off+1): 1}, {(b_row_off+1, b_col_off+1): 1}, {(c_col_off + 1)*n + c_row: 1})
    
    # 2x2x2 blocks (Strassen, rank 7 each): 6 blocks
    # C[0:2, 0:2] = A[0:2,0:2]×B[0:2,0:2] + A[0:2,2:4]×B[2:4,0:2]
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)
    
    # C[0:2, 2:4] = A[0:2,0:2]×B[0:2,2:4] + A[0:2,2:4]×B[2:4,2:4]
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)
    
    # C[0:2, 4:6] = A[0:2,0:2]×B[0:2,4:6] + A[0:2,2:4]×B[2:4,4:6]
    add_strassen_2x2x2(0, 0, 0, 4, 0, 4)
    add_strassen_2x2x2(0, 2, 2, 4, 0, 4)
    
    # 1x2x2 blocks (schoolbook, rank 4 each): 6 blocks
    # C[2:3, 0:2] = A[2:3,0:2]×B[0:2,0:2] + A[2:3,2:4]×B[2:4,0:2]
    add_schoolbook_1x2x2(2, 0, 0, 0, 2, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0, 2, 0)
    
    # C[2:3, 2:4] = A[2:3,0:2]×B[0:2,2:4] + A[2:3,2:4]×B[2:4,2:4]
    add_schoolbook_1x2x2(2, 0, 0, 2, 2, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2, 2, 2)
    
    # C[2:3, 4:6] = A[2:3,0:2]×B[0:2,4:6] + A[2:3,2:4]×B[2:4,4:6]
    add_schoolbook_1x2x2(2, 0, 0, 4, 2, 4)
    add_schoolbook_1x2x2(2, 2, 2, 4, 2, 4)
    
    # Convert to matrix format
    rank = len(U)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for (i, j), val in U[r].items():
            U_mat[i * m + j][r] = val
        for (j, k), val in V[r].items():
            V_mat[j * p + k][r] = val
        for idx, val in W[r].items():
            W_mat[idx][r] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
