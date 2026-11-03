import sys
import time

def construct():
    """
    Rank-50 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    Uses Strassen (rank 7) for 2x2x2 blocks and schoolbook (rank 4) for 2x2x1 blocks.
    Total: 6*7 + 2*4 = 42 + 8 = 50.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = 2, 4, 7
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_indices, B_indices, C_indices):
        nonlocal col
        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            nonlocal col
            for local_a, coeff in A_coeffs.items():
                global_a = A_indices[local_a]
                U[global_a[0] * m + global_a[1]][col] += coeff
            for local_b, coeff in B_coeffs.items():
                global_b = B_indices[local_b]
                V[global_b[0] * p + global_b[1]][col] += coeff
            for local_c, coeff in C_coeffs.items():
                global_c = C_indices[local_c]
                W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        # M1: (A00+A11)(B00+B11) -> C00+C11
        process_mk({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        # M2: (A10+A11)(B00) -> C10, C11 -= M2
        process_mk({2:1, 3:1}, {0:1}, {2:1, 3:-1})
        # M3: (A00)(B11-B01) -> C01, C11 += M3
        process_mk({0:1}, {1:1, 3:-1}, {1:1, 3:1})
        # M4: (A11)(B01-B00) -> C00, C10 += M4
        process_mk({3:1}, {2:1, 0:-1}, {0:1, 2:1})
        # M5: (A00+A01)(B11) -> C01, C00 -= M5
        process_mk({0:1, 1:1}, {3:1}, {0:-1, 1:1})
        # M6: (A10+A00)(B00+B01) -> C11
        process_mk({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        # M7: (A01-A10)(B10+B11) -> C10
        process_mk({1:1, 3:-1}, {2:1, 3:1}, {0:1})
        
    def add_schoolbook_221(A_indices, B_indices, C_indices):
        nonlocal col
        def process_mult(A_idx, B_idx, C_idx, coeff=1):
            nonlocal col
            global_a = A_indices[A_idx]
            global_b = B_indices[B_idx]
            global_c = C_indices[C_idx]
            
            U[global_a[0] * m + global_a[1]][col] += coeff
            V[global_b[0] * p + global_b[1]][col] += coeff
            W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        # 2x2 * 2x1 -> 2x1: 4 scalar multiplications
        process_mult(0, 0, 0, 1)  # A[0,0]*B[0,0] -> C[0,0]
        process_mult(1, 1, 0, 1)  # A[0,1]*B[1,0] -> C[0,0]
        process_mult(2, 0, 1, 1)  # A[1,0]*B[0,0] -> C[1,0]
        process_mult(3, 1, 1, 1)  # A[1,1]*B[1,0] -> C[1,0]
        
    # A split: A_L = rows 0,1 of A (indices (0,0),(0,1),(1,0),(1,1))
    #           A_R = rows 2,3 of A (indices (0,2),(0,3),(1,2),(1,3))
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    
    # Block 1: B cols 0,1 (2x2 sub-blocks)
    B_1_L = [(0,0), (0,1), (1,0), (1,1)]
    B_1_R = [(2,0), (2,1), (3,0), (3,1)]
    C_1 = [(0,0), (0,1), (1,0), (1,1)]
    
    if time.time() < deadline:
        add_strassen_block(A_L, B_1_L, C_1)  # rank 7
        add_strassen_block(A_R, B_1_R, C_1)  # rank 7
    
    # Block 2: B cols 2,3 (2x2 sub-blocks)
    B_2_L = [(0,2), (0,3), (1,2), (1,3)]
    B_2_R = [(2,2), (2,3), (3,2), (3,3)]
    C_2 = [(0,2), (0,3), (1,2), (1,3)]
    
    if time.time() < deadline:
        add_strassen_block(A_L, B_2_L, C_2)  # rank 7
        add_strassen_block(A_R, B_2_R, C_2)  # rank 7
    
    # Block 3: B cols 4,5 (2x2 sub-blocks)
    B_3_L_L = [(0,4), (0,5), (1,4), (1,5)]
    B_3_L_R = [(2,4), (2,5), (3,4), (3,5)]
    C_3_L = [(0,4), (0,5), (1,4), (1,5)]
    
    if time.time() < deadline:
        add_strassen_block(A_L, B_3_L_L, C_3_L)  # rank 7
        add_strassen_block(A_R, B_3_L_R, C_3_L)  # rank 7
    
    # Block 4: B col 6 (2x1 sub-blocks)
    B_4_L = [(0,6), (1,6)]
    B_4_R = [(2,6), (3,6)]
    C_4 = [(0,6), (1,6)]
    
    if time.time() < deadline:
        add_schoolbook_221(A_L, B_4_L, C_4)  # rank 4
        add_schoolbook_221(A_R, B_4_R, C_4)  # rank 4
    
    return {"U": U, "V": V, "W": W}
