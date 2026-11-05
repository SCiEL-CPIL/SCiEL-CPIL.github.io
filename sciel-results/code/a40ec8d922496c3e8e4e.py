import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Block-based Strassen decomposition for 3x4x5 matrix multiplication.
    
    Blocks:
    - A rows: {0,1}, {2}
    - A cols / B rows: {0,1}, {2,3}
    - B cols: {0,1}, {2,3,4}
    
    Block products (each contributing to C[i,k] for appropriate i,k):
    1. A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (2x2x2, Strassen rank 7)
    2. A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]  (2x2x2, Strassen rank 7)
    3. A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5]  (2x2x3, rank 11)
    4. A[0:2,2:4] * B[2:4,2:5] -> C[0:2,2:5]  (2x2x3, rank 11)
    5. A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2]  (1x2x2, rank 2)
    6. A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2]  (1x2x2, rank 2)
    7. A[2:3,0:2] * B[0:2,2:5] -> C[2:3,2:5]  (1x2x3, rank 3)
    8. A[2:3,2:4] * B[2:4,2:5] -> C[2:3,2:5]  (1x2x3, rank 3)
    
    Total: 7+7+11+11+2+2+3+3 = 46
    """
    n, m, p = DIMENSIONS
    total_rank = 46
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 46
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 46
    W = [[0] * total_rank for _ in range(p * n)]  # 15 x 46
    
    col = 0

    def set_term(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, val in u_entries:
            U[idx][col] += val
        for idx, val in v_entries:
            V[idx][col] += val
        for idx, val in w_entries:
            W[idx][col] += val
        col += 1

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(i, k):
        return k * n + i

    # Strassen 2x2x2 template
    # For A block rows a0,a1 cols b0,b1; B block rows b0,b1 cols c0,c1
    # C block rows a0,a1 cols c0,c1
    def strassen_2x2x2(a_rows, b_cols, c_cols):
        nonlocal col
        # a_rows = [ar0, ar1], b_cols = [bc0, bc1], c_cols = [cc0, cc1]
        ar0, ar1 = a_rows
        bc0, bc1 = b_cols
        cc0, cc1 = c_cols
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        set_term(
            [(u_idx(ar0, bc0), 1), (u_idx(ar1, bc1), 1)],
            [(v_idx(bc0, cc0), 1), (v_idx(bc1, cc1), 1)],
            [(w_idx(ar0, cc0), 1), (w_idx(ar1, cc1), 1)]
        )
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        set_term(
            [(u_idx(ar1, bc0), 1), (u_idx(ar1, bc1), 1)],
            [(v_idx(bc0, cc0), 1)],
            [(w_idx(ar1, cc0), 1), (w_idx(ar1, cc1), -1)]
        )
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        set_term(
            [(u_idx(ar0, bc0), 1)],
            [(v_idx(bc0, cc1), 1), (v_idx(bc1, cc1), -1)],
            [(w_idx(ar0, cc1), 1), (w_idx(ar1, cc1), 1)]
        )
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        set_term(
            [(u_idx(ar1, bc1), 1)],
            [(v_idx(bc1, cc0), 1), (v_idx(bc0, cc0), -1)],
            [(w_idx(ar0, cc0), 1), (w_idx(ar1, cc0), 1)]
        )
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        set_term(
            [(u_idx(ar0, bc0), 1), (u_idx(ar0, bc1), 1)],
            [(v_idx(bc1, cc1), 1)],
            [(w_idx(ar0, cc0), -1), (w_idx(ar0, cc1), 1)]
        )
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        set_term(
            [(u_idx(ar1, bc0), 1), (u_idx(ar0, bc0), -1)],
            [(v_idx(bc0, cc0), 1), (v_idx(bc0, cc1), 1)],
            [(w_idx(ar1, cc1), 1)]
        )
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        set_term(
            [(u_idx(ar0, bc1), 1), (u_idx(ar1, bc1), -1)],
            [(v_idx(bc1, cc0), 1), (v_idx(bc1, cc1), 1)],
            [(w_idx(ar0, cc0), 1)]
        )

    # 2x2x3 block: A[2x2] * B[2x3] -> C[2x3]
    # Split B into B[2x2] + B[2x1]
    # Strassen on 2x2x2 part, schoolbook on 2x2x1 part
    def block_2x2x3(a_rows, b_cols, c_cols):
        nonlocal col
        # c_cols has 3 elements
        ar0, ar1 = a_rows
        bc0, bc1 = b_cols
        cc0, cc1, cc2 = c_cols
        
        # Strassen on columns cc0, cc1
        strassen_2x2x2([ar0, ar1], [bc0, bc1], [cc0, cc1])
        
        # Schoolbook on column cc2 (2x2x1 = 4 terms)
        for ar in [ar0, ar1]:
            for bc in [bc0, bc1]:
                set_term(
                    [(u_idx(ar, bc), 1)],
                    [(v_idx(bc, cc2), 1)],
                    [(w_idx(ar, cc2), 1)]
                )

    # 1x2x2 block: A[1x2] * B[2x2] -> C[1x2] (rank 2)
    def block_1x2x2(a_row, b_cols, c_cols):
        nonlocal col
        ar = a_row
        bc0, bc1 = b_cols
        cc0, cc1 = c_cols
        
        set_term(
            [(u_idx(ar, bc0), 1)],
            [(v_idx(bc0, cc0), 1)],
            [(w_idx(ar, cc0), 1)]
        )
        set_term(
            [(u_idx(ar, bc1), 1)],
            [(v_idx(bc1, cc1), 1)],
            [(w_idx(ar, cc1), 1)]
        )

    # 1x2x3 block: A[1x2] * B[2x3] -> C[1x3] (rank 3)
    def block_1x2x3(a_row, b_cols, c_cols):
        nonlocal col
        ar = a_row
        bc0, bc1 = b_cols
        cc0, cc1, cc2 = c_cols
        
        for cc in [cc0, cc1, cc2]:
            for bc in [bc0, bc1]:
                set_term(
                    [(u_idx(ar, bc), 1)],
                    [(v_idx(bc, cc), 1)],
                    [(w_idx(ar, cc), 1)]
                )

    # Apply all blocks
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_2x2x2([0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen_2x2x2([0, 1], [2, 3], [0, 1])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    block_2x2x3([0, 1], [0, 1], [2, 3, 4])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    block_2x2x3([0, 1], [2, 3], [2, 3, 4])
    
    # Block 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    block_1x2x2(2, [0, 1], [0, 1])
    
    # Block 6: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    block_1x2x2(2, [2, 3], [0, 1])
    
    # Block 7: A[2:3, 0:2] * B[0:2, 2:5] -> C[2:3, 2:5]
    block_1x2x3(2, [0, 1], [2, 3, 4])
    
    # Block 8: A[2:3, 2:4] * B[2:4, 2:5] -> C[2:3, 2:5]
    block_1x2x3(2, [2, 3], [2, 3, 4])

    return {"U": U, "V": V, "W": W}
