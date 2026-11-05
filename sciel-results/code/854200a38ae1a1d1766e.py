"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-structured decomposition for 6x6 matrix multiplication.
    Partition 6x6 into a 2x2 grid of 3x3 blocks.
    Use Strassen algorithm for the 2x2 block-level multiplication (rank 7).
    Use Schoolbook algorithm for the 3x3 block-level multiplication (rank 27).
    Total rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block size is 3, number of blocks in each dimension is 2
    bs = 3
    nb = 2  # number of blocks per dimension
    
    # Strassen for 2x2 block matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen term t, we have:
    # U part: linear combo of A blocks
    # V part: linear combo of B blocks
    # W part: linear combo of C blocks
    
    # A block indices: 00, 01, 10, 11
    # B block indices: 00, 01, 10, 11
    # C block indices: 00, 01, 10, 11
    
    # Strassen coefficients for A blocks (4 blocks)
    # t: 0..6
    strassen_A_blocks = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    # Strassen coefficients for B blocks (4 blocks)
    strassen_B_blocks = [
        [1, 1, 0, 0], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10 - B11
        [-1, 1, 0, 0],# M4: B01 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 0, 1, 0], # M6: B00 + B10
        [0, 0, 1, 1]  # M7: B11 + B01
    ]
    
    # Output coefficients for C blocks (4 blocks)
    # t: 0..6
    output_coeffs_blocks = [
        [1, 0, 0, 1],   # M1: C00=1, C01=0, C10=0, C11=1
        [0, 0, 1, -1],  # M2: C00=0, C01=0, C10=1, C11=-1
        [0, 1, 0, 1],   # M3: C00=0, C01=1, C10=0, C11=1
        [1, 0, 1, 0],   # M4: C00=1, C01=0, C10=1, C11=0
        [-1, 1, 0, 0],  # M5: C00=-1, C01=1, C10=0, C11=0
        [0, 0, 0, 1],   # M6: C00=0, C01=0, C10=0, C11=1
        [1, 0, 0, 0]    # M7: C00=1, C01=0, C10=0, C11=0
    ]
    
    # Total rank: 7 (Strassen) * 27 (Schoolbook for 3x3)
    total_rank = 7 * 27
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Schoolbook for 3x3 matrix multiplication
    # A 3x3, B 3x3 -> C 3x3
    # 27 terms: for each i,j in 3x3, C_ij = sum_k A_ik B_kj
    # We iterate over the 27 schoolbook terms for each Strassen term
    
    for t in range(7):
        # For each of the 27 schoolbook multiplications within the 3x3 blocks
        for i_block in range(3): # row of A block
            for j_block in range(3): # col of A block
                for k_block in range(3): # row of B block / col of A block
                
                    # This corresponds to C_local[i_block, j_block] += A_local[i_block, k_block] * B_local[k_block, j_block]
                    
                    # Determine which global block this local entry belongs to
                    # A block index: determined by strassen_A_blocks[t]
                    # B block index: determined by strassen_B_blocks[t]
                    
                    # We need to add contributions to U, V, W for this specific scalar multiplication
                    # U index: row in A, col in A
                    # V index: row in B, col in B
                    # W index: row in C, col in C
                    
                    # The local indices i_block, k_block, j_block are within the 3x3 block.
                    # The global block indices are determined by the Strassen coefficients.
                    
                    # For U:
                    # We are multiplying A_block[i_block, k_block]
                    # Which A block? It depends on the Strassen term t.
                    # strassen_A_blocks[t] gives coefficients for A00, A01, A10, A11.
                    # But we are doing schoolbook *inside* the blocks.
                    # So for a given Strassen term t, we are effectively computing:
                    # M_t = (sum_a coeff_A_t[a] * A_a) * (sum_b coeff_B_t[b] * B_b)
                    # And we expand this product using schoolbook.
                    # This means we are doing 27 scalar multiplications for each term t.
                    # Each scalar multiplication is:
                    # (linear combo of A entries in specific blocks) * (linear combo of B entries in specific blocks)
                    
                    # Wait, the Strassen coefficients are for the *blocks*.
                    # So for term t, the "A matrix" is a linear combination of the 4 A blocks.
                    # The "B matrix" is a linear combination of the 4 B blocks.
                    # The product is a 3x3 matrix (since blocks are 3x3).
                    # We compute this product using schoolbook (27 multiplications).
                    
                    # So for each t, and for each local (i, k, j) in 3x3:
                    # We need to form the linear combination of A entries for A_local[i,k]
                    # and B entries for B_local[k,j].
                    
                    a_coeff_block = None
                    b_coeff_block = None
                    
                    # Actually, the Strassen coefficients apply to the whole block.
                    # So A_local[i,k] in the "combined" matrix is:
                    # sum_{a_idx} strassen_A_blocks[t][a_idx] * A_block[a_idx][i,k]
                    
                    # Similarly for B.
                    
                    # So for this specific scalar multiplication (i, k, j):
                    # U contributions:
                    # For each A block a_idx with coeff c_a:
                    #   U[global_row(i, a_idx_row, a_idx_col), global_col(k, a_idx_row, a_idx_col)] += c_a
                    
                    # V contributions:
                    # For each B block b_idx with coeff c_b:
                    #   V[global_row(k, b_idx_row, b_idx_col), global_col(j, b_idx_row, b_idx_col)] += c_b
                    
                    # W contributions:
                    # The result contributes to C_local[i,j].
                    # C_local[i,j] in the "combined" matrix is:
                    # sum_{c_idx} output_coeffs_blocks[t][c_idx] * C_block[c_idx][i,j]
                    # Wait, the Strassen output gives C blocks.
                    # So C_block[c_idx][i,j] gets contribution from M_t.
                    # And M_t is the product of the combined A and combined B.
                    # So C_block[c_idx][i,j] += output_coeffs_blocks[t][c_idx] * (A_combined[i,k] * B_combined[k,j])
                    
                    # So for W:
                    # For each C block c_idx with coeff c_c:
                    #   W[global_row(i, c_idx_row, c_idx_col), global_col(j, c_idx_row, c_idx_col)] += c_c
                    
                    # Let's implement this.
                    
                    # A block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # B block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # C block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    
                    a_blocks = [(0,0), (0,1), (1,0), (1,1)]
                    b_blocks = [(0,0), (0,1), (1,0), (1,1)]
                    c_blocks = [(0,0), (0,1), (1,0), (1,1)]
                    
                    # U
                    for a_idx in range(4):
                        coeff = strassen_A_blocks[t][a_idx]
                        if coeff != 0:
                            br, bc = a_blocks[a_idx]
                            # Global row: br * bs + i_block
                            # Global col: bc * bs + k_block
                            u_row = br * bs + i_block
                            u_col = bc * bs + k_block
                            U[u_row * m + u_col][col] = coeff
                    
                    # V
                    for b_idx in range(4):
                        coeff = strassen_B_blocks[t][b_idx]
                        if coeff != 0:
                            br, bc = b_blocks[b_idx]
                            # Global row: br * bs + k_block
                            # Global col: bc * bs + j_block
                            v_row = br * bs + k_block
                            v_col = bc * bs + j_block
                            V[v_row * p + v_col][col] = coeff
                    
                    # W
                    for c_idx in range(4):
                        coeff = output_coeffs_blocks[t][c_idx]
                        if coeff != 0:
                            br, bc = c_blocks[c_idx]
                            # Global row: br * bs + i_block
                            # Global col: bc * bs + j_block
                            # W index: col_C * n + row_C
                            w_row = br * bs + i_block
                            w_col = bc * bs + j_block
                            W[w_col * n + w_row][col] = coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
