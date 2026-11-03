import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 114 decomposition
    rank = 114
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term for 2x2 blocks
    # a_block: (row_offset, col_offset) for A
    # b_block: (row_offset, col_offset) for B
    # c_block: (row_offset, col_offset) for C
    # u, v, w: 2x2 matrices (lists of lists)
    def add_term(a_off, b_off, c_off, u, v, w, col_idx):
        for i in range(2):
            for j in range(2):
                # A index
                ai = a_off[0] + i
                aj = a_off[1] + j
                U[ai * m + aj][col_idx] += u[i][j]
                
                # B index
                bi = b_off[0] + i
                bj = b_off[1] + j
                V[bi * p + bj][col_idx] += v[i][j]
                
                # C index
                ci = c_off[0] + i
                cj = c_off[1] + j
                W[cj * n + ci][col_idx] += w[i][j]

    # Small Strassen 2x2 steps
    # u, v, w are 2x2
    small_steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # p1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # p2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # p3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # p4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # p5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # p6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # p7
    ]
    
    # Convert flat lists to 2x2 for clarity in add_term if needed, or just index
    def to_2x2(flat):
        return [[flat[0], flat[1]], [flat[2], flat[3]]]

    # Big Strassen 4x4 steps
    # Each step defines A blocks, B blocks, C blocks
    # Format: list of (a_off, b_off, c_off) pairs? No.
    # It's (sum of A blocks) * (sum of B blocks) -> sum of C blocks.
    # We expand the product.
    
    # Step 1: (A11 + A22)(B11 + B22) -> C11 + C22
    # Pairs: (A11, B11), (A11, B22), (A22, B11), (A22, B22)
    # All contribute to C11 and C22?
    # Wait. The product M1 is a 2x2 matrix.
    # M1 is added to C11 and C22.
    # So for each (A_blk, B_blk) pair, we compute the 2x2 product P.
    # Then P is added to C11 and C22.
    
    def process_big_step(a_blocks, b_blocks, c_blocks, col_start):
        # a_blocks: list of (offset, sign)
        # b_blocks: list of (offset, sign)
        # c_blocks: list of (offset, sign)
        
        # Iterate over all pairs of A and B blocks
        for a_off, a_sign in a_blocks:
            for b_off, b_sign in b_blocks:
                # For each of the 7 small steps
                for u_flat, v_flat, w_flat in small_steps:
                    u = to_2x2(u_flat)
                    v = to_2x2(v_flat)
                    w = to_2x2(w_flat)
                    
                    # Scale u and v by signs
                    for i in range(2):
                        for j in range(2):
                            u[i][j] *= a_sign
                            v[i][j] *= b_sign
                    
                    # Add to U, V
                    # W will be added for each C block
                    # First, update U and V
                    for i in range(2):
                        for j in range(2):
                            ai = a_off[0] + i
                            aj = a_off[1] + j
                            U[ai * m + aj][col_start] += u[i][j]
                            
                            bi = b_off[0] + i
                            bj = b_off[1] + j
                            V[bi * p + bj][col_start] += v[i][j]
                    
                    # Update W for each C block
                    for c_off, c_sign in c_blocks:
                        for i in range(2):
                            for j in range(2):
                                ci = c_off[0] + i
                                cj = c_off[1] + j
                                W[cj * n + ci][col_start] += w[i][j] * c_sign
                    
                    col_start += 1
        return col_start

    # Define blocks for 4x4
    # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
    A11 = (0, 0)
    A12 = (0, 2)
    A21 = (2, 0)
    A22 = (2, 2)
    
    B11 = (0, 0)
    B12 = (0, 2)
    B21 = (2, 0)
    B22 = (2, 2)
    
    C11 = (0, 0)
    C12 = (0, 2)
    C21 = (2, 0)
    C22 = (2, 2)
    
    # Step 1
    col = process_big_step(
        [(A11, 1), (A22, 1)],
        [(B11, 1), (B22, 1)],
        [(C11, 1), (C22, 1)],
        col
    )
    
    # Step 2: (A21 + A22) B11 -> C21 + C22 (with -1 for C22)
    # C21 = M2 + M4. C22 = M1 - M2 + ...
    col = process_big_step(
        [(A21, 1), (A22, 1)],
        [(B11, 1)],
        [(C21, 1), (C22, -1)],
        col
    )
    
    # Step 3: A11 (B12 - B22) -> C12 + C22
    # C12 = M3 + M5. C22 = ... + M3
    col = process_big_step(
        [(A11, 1)],
        [(B12, 1), (B22, -1)],
        [(C12, 1), (C22, 1)],
        col
    )
    
    # Step 4: A22 (B21 - B11) -> C11 + C21
    # C11 = ... + M4. C21 = M2 + M4
    col = process_big_step(
        [(A22, 1)],
        [(B21, 1), (B11, -1)],
        [(C11, 1), (C21, 1)],
        col
    )
    
    # Step 5: (A11 + A12) B22 -> C11 (with -1) + C12
    # C11 = ... - M5. C12 = M3 + M5
    col = process_big_step(
        [(A11, 1), (A12, 1)],
        [(B22, 1)],
        [(C11, -1), (C12, 1)],
        col
    )
    
    # Step 6: (A21 - A11) (B11 + B12) -> C22
    # C22 = ... + M6
    col = process_big_step(
        [(A21, 1), (A11, -1)],
        [(B11, 1), (B12, 1)],
        [(C22, 1)],
        col
    )
    
    # Step 7: (A12 - A22) (B21 + B22) -> C11
    # C11 = ... + M7
    col = process_big_step(
        [(A12, 1), (A22, -1)],
        [(B21, 1), (B22, 1)],
        [(C11, 1)],
        col
    )
    
    # Now handle the 4x1 part (last column of B)
    # A (4x4) * v (4x1) -> C (4x1)
    # v is column 8 (index 8) of B.
    # C is column 8 (index 8) of C.
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
