import time
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-17 decomposition for 2x3 * 3x3 matrix multiplication.
    Uses Strassen for the 2x2x2 core, schoolbook for the 2x2x1 strip,
    and outer product for the remaining 2x1x3 part.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = 17

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to set a column
    def set_col(col_idx, u_indices, v_indices, w_indices):
        for idx in u_indices:
            U[idx][col_idx] = 1
        for idx in v_indices:
            V[idx][col_idx] = 1
        for idx, val in w_indices:
            W[idx][col_idx] = val

    # 1. Strassen for M_2,2,2 (cols 0,1 of B, rows 0,1 of A, cols 0,1 of C)
    # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # B indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # C indices: 0(0,0), 1(1,0), 2(0,1), 3(1,1)
    
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    set_col(0, [0, 4], [0, 4], [(0, 1), (3, 1)])
    
    # M2: (A10+A11)B00 -> C10+=1, C11-=1
    set_col(1, [3, 4], [0], [(1, 1), (3, -1)])
    
    # M3: A00(B01-B11) -> C01+=1, C11+=1
    set_col(2, [0], [1, 4], [(2, 1), (3, 1)])
    # Note: B11 index is 4. Value is -1.
    V[4][2] = -1
    
    # M4: A11(B10-B00) -> C00+=1, C10+=1
    set_col(3, [4], [3, 0], [(0, 1), (1, 1)])
    # Note: B00 index is 0. Value is -1.
    V[0][3] = -1
    
    # M5: (A00+A10)B11 -> C00-=1, C01+=1
    set_col(4, [0, 3], [4], [(0, -1), (2, 1)])
    
    # M6: (A11-A00)(B00+B01) -> C11+=1
    set_col(5, [4, 0], [0, 1], [(3, 1)])
    # Note: A00 index is 0. Value is -1.
    U[0][5] = -1
    
    # M7: (A01-A11)(B10+B11) -> C00+=1
    set_col(6, [1, 4], [3, 4], [(0, 1)])
    # Note: A11 index is 4. Value is -1.
    U[4][6] = -1

    # 2. Schoolbook for M_2,2,1 (cols 0,1 of A, col 2 of B, col 2 of C)
    # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # B indices: 2(0,2), 5(1,2)
    # C indices: 4(0,2), 5(1,2)
    
    # C02 += A00*B02
    set_col(7, [0], [2], [(4, 1)])
    
    # C02 += A01*B12
    set_col(8, [1], [5], [(4, 1)])
    
    # C12 += A10*B02
    set_col(9, [3], [2], [(5, 1)])
    
    # C12 += A11*B12
    set_col(10, [4], [5], [(5, 1)])

    # 3. Outer product for A_col2 * B_row2 (row 2 of A, row 2 of B, all C)
    # A indices: 2(0,2), 5(1,2)
    # B indices: 6(2,0), 7(2,1), 8(2,2)
    # C indices: 0(0,0), 2(0,1), 4(0,2), 1(1,0), 3(1,1), 5(1,2)
    
    # C00 += A02*B20
    set_col(11, [2], [6], [(0, 1)])
    
    # C01 += A02*B21
    set_col(12, [2], [7], [(2, 1)])
    
    # C02 += A02*B22
    set_col(13, [2], [8], [(4, 1)])
    
    # C10 += A12*B20
    set_col(14, [5], [6], [(1, 1)])
    
    # C11 += A12*B21
    set_col(15, [5], [7], [(3, 1)])
    
    # C12 += A12*B22
    set_col(16, [5], [8], [(5, 1)])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
