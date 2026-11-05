import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # For 6x6 matrix multiplication, we can use a block Strassen approach.
    # Partition 6x6 into four 3x3 blocks.
    # Use Strassen for the 2x2 block structure, and schoolbook for each 3x3 multiplication.
    # This gives rank = 7 * 27 = 189.
    
    # But we can do better by using a 3x3 block structure with 2x2 Strassen.
    # Partition 6x6 into nine 2x2 blocks.
    # Use Strassen for each pair of 2x2 blocks? No, that's not quite right.
    
    # Actually, let's think about this differently.
    # The standard approach for n x n matrix multiplication using Strassen recursively:
    # R(n) = 7 * R(n/2) for n even, R(1) = 1
    # R(6) = 7 * R(3)
    # R(3) = 7 * R(1.5) -- not integer, so this doesn't work directly.
    
    # Alternative: Use 2x2 Strassen blocks.
    # Partition 6x6 into a 3x3 grid of 2x2 blocks.
    # For each 2x2 block multiplication, use Strassen (rank 7).
    # But the blocks interact...
    
    # Let me just implement the 2x2 Strassen with 3x3 schoolbook kernels.
    # This gives rank 189, which is better than schoolbook 216.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A-blocks: A11 (rows 0-2, cols 0-2), A12 (rows 0-2, cols 3-5), 
    #           A21 (rows 3-5, cols 0-2), A22 (rows 3-5, cols 3-5)
    # B-blocks: B11 (rows 0-2, cols 0-2), B12 (rows 0-2, cols 3-5),
    #           B21 (rows 3-5, cols 0-2), B22 (rows 3-5, cols 3-5)
    
    a_coeffs = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1], # M7: A12 - A22
    ]
    
    b_coeffs = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12 - B22
        [0, 0, 1, -1], # M4: B21 - B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [0, 0, 1, 1],  # M7: B21 + B22
    ]
    
    c_block_coeffs = {
        0: {0: 1, 1: 0, 2: 0, 3: 1},  # M1 -> C11(1), C22(1)
        1: {0: 0, 1: 0, 2: 1, 3: -1},  # M2 -> C21(1), C22(-1)
        2: {0: 0, 1: 1, 2: 0, 3: 1},   # M3 -> C12(1), C22(1)
        3: {0: 1, 1: 0, 2: 1, 3: 0},   # M4 -> C11(1), C21(1)
        4: {0: -1, 1: 1, 2: 0, 3: 0},  # M5 -> C11(-1), C12(1)
        5: {0: 0, 1: 0, 2: 0, 3: 1},   # M6 -> C22(1)
        6: {0: 1, 1: 0, 2: 0, 3: 0},   # M7 -> C11(1)
    }
    
    for t in range(7):
        for i_sub in range(3):
            for k_sub in range(3):
                for j_sub in range(3):
                    col = t * 27 + i_sub * 9 + k_sub * 3 + j_sub
                    
                    # A-element: A_comb_t[i_sub, k_sub]
                    for a_block in range(4):
                        coeff_a = a_coeffs[t][a_block]
                        if coeff_a == 0:
                            continue
                        
                        if a_block == 0:  # A11
                            i_global = i_sub
                            j_global = k_sub
                        elif a_block == 1:  # A12
                            i_global = i_sub
                            j_global = 3 + k_sub
                        elif a_block == 2:  # A21
                            i_global = 3 + i_sub
                            j_global = k_sub
                        else:  # a_block == 3, A22
                            i_global = 3 + i_sub
                            j_global = 3 + k_sub
                        
                        U[i_global * m + j_global][col] = coeff_a
                    
                    # B-element: B_comb_t[k_sub, j_sub]
                    for b_block in range(4):
                        coeff_b = b_coeffs[t][b_block]
                        if coeff_b == 0:
                            continue
                        
                        if b_block == 0:  # B11
                            i_global = k_sub
                            j_global = j_sub
                        elif b_block == 1:  # B12
                            i_global = k_sub
                            j_global = 3 + j_sub
                        elif b_block == 2:  # B21
                            i_global = 3 + k_sub
                            j_global = j_sub
                        else:  # b_block == 3, B22
                            i_global = 3 + k_sub
                            j_global = 3 + j_sub
                        
                        V[i_global * p + j_global][col] = coeff_b
                    
                    # C-element
                    for c_block in range(4):
                        coeff_c = c_block_coeffs[t][c_block]
                        if coeff_c == 0:
                            continue
                        
                        if c_block == 0:  # C11
                            i_global = i_sub
                            j_global = j_sub
                        elif c_block == 1:  # C12
                            i_global = i_sub
                            j_global = 3 + j_sub
                        elif c_block == 2:  # C21
                            i_global = 3 + i_sub
                            j_global = j_sub
                        else:  # c_block == 3, C22
                            i_global = 3 + i_sub
                            j_global = 3 + j_sub
                        
                        # W[k*n + i, col] where i is row, k is col of C
                        W[j_global * n + i_global][col] = coeff_c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
