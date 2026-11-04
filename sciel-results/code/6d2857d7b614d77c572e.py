import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    _t0 = time.time()
    n, m, p = 4, 4, 6

    # Strategy: 
    # 1. Compute the 4x4 * 4x4 part using recursive Strassen (rank 49).
    # 2. Compute the 4x4 * 4x2 part using block Strassen (rank 28).
    #    - Split B's last 2 columns into a 4x2 block.
    #    - Split A into 2x2 blocks (4 blocks).
    #    - Split B_block into 2x2 blocks (2 blocks: top 2 rows, bottom 2 rows).
    #    - For each of the 4 A-blocks and 2 B-blocks, use 2x2 Strassen (rank 7).
    #    - Total rank for this part: 4 * 2 * 7 = 56? No, wait.
    #    Let's re-verify the block structure for 4x4 * 4x2.
    #    A is 4x4. B is 4x2.
    #    A = [A00 A01; A10 A11] where each is 2x2.
    #    B = [B0; B1] where B0 is 2x2, B1 is 2x2.
    #    C = A * B = [A00*B0 + A01*B1; A10*B0 + A11*B1].
    #    This requires 8 products of 2x2 matrices.
    #    Each 2x2 product has rank 7.
    #    Total rank = 8 * 7 = 56.
    #    
    #    Wait, the inspiration says rank 28 for 4x4x2. How?
    #    Maybe it uses a different decomposition.
    #    Let's check the proven best score mentioned: "rank 77".
    #    49 (for 4x4x4) + 28 (for 4x4x2) = 77.
    #    So the 4x4x2 part must have rank 28.
    #    How can 4x4 * 4x2 have rank 28?
    #    Schoolbook for 4x4 * 4x2 is 4*4*2 = 32.
    #    Strassen-like reduction:
    #    If we view 4x4 * 4x2 as two 4x4 * 4x1 problems? No.
    #    Let's look at the structure.
    #    Maybe it splits A into 2x4 and B into 4x2?
    #    (2x4)*(4x2) = (2x2)*(2x2) + (2x2)*(2x2). Rank 7+7=14 per 2x4 block.
    #    There are 2 such blocks (top 2 rows of A, bottom 2 rows of A).
    #    Total rank = 2 * 14 = 28.
    #    Yes! This matches.
    #    
    #    So the plan is:
    #    1. C_left (4x4) = A * B_left (4x4) using recursive Strassen (rank 49).
    #    2. C_right (4x2) = A * B_right (4x2) using the block method (rank 28).
    #       - Split A into A_top (2x4) and A_bot (2x4).
    #       - B_right is 4x2.
    #       - C_top (2x2) = A_top * B_right.
    #         A_top = [A_top_00, A_top_01] (each 2x2).
    #         B_right = [B_top; B_bot] (each 2x2).
    #         C_top = A_top_00 * B_top + A_top_01 * B_bot.
    #         This is sum of two 2x2*2x2 products. Rank 7+7=14.
    #       - C_bot (2x2) = A_bot * B_right.
    #         Similarly, rank 14.
    #       - Total rank for right part = 14 + 14 = 28.
    #    
    #    Total rank = 49 + 28 = 77.

    R = 77
    U = np.zeros((n * m, R))  # (16, 77)
    V = np.zeros((m * p, R))  # (24, 77)
    W = np.zeros((p * n, R))  # (24, 77)

    # --- Part 1: 4x4 * 4x4 using recursive Strassen (rank 49) ---
    # We need the factors for 4x4*4x4.
    # Recursive Strassen:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # Use 7 Strassen multiplications of 2x2*2x2.
    # Each 2x2*2x2 has rank 7.
    # Total rank 7*7 = 49.
    
    # Define 2x2 Strassen factors
    alpha_2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta_2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon_2 = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    # Recursive step: 4x4*4x4
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    # Each Ai, Bi is 2x2.
    # Strassen products P1..P7 of 2x2 matrices.
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    col = 0
    # We will iterate over the 7 Strassen terms for the 4x4 level.
    # For each term, we have a product of two 2x2 matrices (composed of 4x4 entries).
    # Wait, the 2x2 matrices here are actually 2x2 blocks of the 4x4 matrix.
    # So each "entry" in the 2x2 matrix is a 2x2 submatrix of the 4x4 matrix.
    # The Strassen formula applies to the 2x2 blocks.
    # So we need to expand the 2x2 Strassen formula into 4x4 entries.
    
    # Let's define the mapping for the 4x4 recursive Strassen.
    # The 2x2 blocks are:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B.
    
    # For each Strassen term s (0..6):
    # The U part is a linear combination of A's entries.
    # The V part is a linear combination of B's entries.
    # The W part is a linear combination of C's entries.
    
    # Since the 2x2 Strassen factors are defined for 2x2 matrices,
    # and our "2x2 matrices" are actually 2x2 blocks of 2x2 matrices,
    # we need to substitute the 2x2 Strassen factors with the actual entries.
    
    # Let's construct the 49 factors explicitly.
    # There are 7 Strassen terms at the top level.
    # Each term involves a product of two 2x2 matrices (blocks).
    # Each 2x2 matrix product can be decomposed into 7 scalar products (rank 7).
    # So total 7 * 7 = 49 scalar products.
    
    # Let's iterate over the 7 top-level Strassen terms.
    # For each, we identify the two 2x2 block matrices M and N.
    # Then we apply the 2x2 Strassen decomposition to M*N.
    
    # Top-level Strassen terms:
    # s=0: M = A00+A11, N = B00+B11
    # s=1: M = A10+A11, N = B00
    # s=2: M = A00, N = B01-B11
    # s=3: M = A11, N = B10-B00
    # s=4: M = A00+A01, N = B11
    # s=5: M = A10-A00, N = B00+B01
    # s=6: M = A01-A11, N = B10+B11
    
    # We need to express M and N as linear combinations of the 2x2 blocks.
    # Then, for each 2x2 Strassen factor (alpha, beta, recon), we combine them.
    
    # Let's define the blocks as indices in the 4x4 matrix.
    # A blocks:
    # A00: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # A01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # A10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # A11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # B blocks:
    # B00: rows 0,1 cols 0,1 -> indices 0,1,6,7
    # B01: rows 0,1 cols 2,3 -> indices 2,3,8,9
    # B10: rows 2,3 cols 0,1 -> indices 12,13,18,19
    # B11: rows 2,3 cols 2,3 -> indices 14,15,20,21
    
    # C blocks:
    # C00: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # C01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # C10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # C11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # For each top-level term s, we have M and N.
    # M is a 2x2 matrix of 2x2 blocks.
    # N is a 2x2 matrix of 2x2 blocks.
    # We apply 2x2 Strassen to M*N.
    # The 2x2 Strassen factors alpha_2, beta_2, recon_2 are for scalar 2x2 matrices.
    # Here, the "scalars" are 2x2 blocks.
    # So we need to expand the alpha/beta coefficients into the actual 4x4 entries.
    
    # Let's define the block indices for A, B, C.
    # A_block[r][c] where r,c in {0,1}
    # A_block[0][0] = A00, A_block[0][1] = A01, A_block[1][0] = A10, A_block[1][1] = A11
    
    # For a given top-level term s, M and N are linear combinations of A_blocks and B_blocks.
    # Let's represent M and N as 2x2 matrices of coefficients for the blocks.
    # e.g., for s=0: M = A00 + A11 -> M[0][0] = A00, M[0][1] = 0, M[1][0] = 0, M[1][1] = A11.
    
    # We will iterate over the 7 top-level terms.
    # For each, we determine the 2x2 matrix of block coefficients for M and N.
    # Then we apply the 2x2 Strassen decomposition.
    
    # Block coefficient matrices for M and N for each top-level term:
    # s=0: M = [[A00, 0], [0, A11]], N = [[B00, 0], [0, B11]]
    # s=1: M = [[0, 0], [A10, A11]], N = [[B00, 0], [0, 0]]
    # s=2: M = [[A00, 0], [0, 0]], N = [[0, B01], [0, -B11]]
    # s=3: M = [[0, 0], [0, A11]], N = [[-B00, 0], [B10, 0]]
    # s=4: M = [[A00, A01], [0, 0]], N = [[0, 0], [0, B11]]
    # s=5: M = [[-A00, 0], [A10, 0]], N = [[B00, B01], [0, 0]]
    # s=6: M = [[0, A01], [0, -A11]], N = [[0, 0], [B10, B11]]
    
    # Let's define these as lists of (block_row, block_col, coeff) for M and N.
    # block_row, block_col in {0,1}
    
    M_coeffs = [
        [(0,0,1), (1,1,1)],  # s=0
        [(1,0,1), (1,1,1)],  # s=1
        [(0,0,1)],           # s=2
        [(1,1,1)],           # s=3
        [(0,0,1), (0,1,1)],  # s=4
        [(0,0,-1), (1,0,1)], # s=5
        [(0,1,1), (1,1,-1)]  # s=6
    ]
    N_coeffs = [
        [(0,0,1), (1,1,1)],  # s=0
        [(0,0,1)],           # s=1
        [(0,1,1), (1,1,-1)], # s=2
        [(0,0,-1), (1,0,1)], # s=3
        [(1,1,1)],           # s=4
        [(0,0,1), (0,1,1)],  # s=5
        [(1,0,1), (1,1,1)]   # s=6
    ]
    
    # Now, for each top-level term s (0..6):
    # We have M_coeffs[s] and N_coeffs[s].
    # We apply the 2x2 Strassen decomposition to the product of the two 2x2 block matrices.
    # The 2x2 Strassen factors are alpha_2, beta_2, recon_2.
    # For each 2x2 Strassen factor t (0..6):
    # alpha_2[t] is a 2x2 matrix of coefficients for the entries of M.
    # beta_2[t] is a 2x2 matrix of coefficients for the entries of N.
    # recon_2[t] gives the output entries.
    
    # The entries of M are the blocks A00, A01, A10, A11.
    # So alpha_2[t][r][c] multiplies the block M[r][c].
    # M[r][c] is a linear combination of A_blocks given by M_coeffs[s].
    # So the total coefficient for A_block[i][j] is sum over r,c of alpha_2[t][r][c] * (coeff of A_block[i][j] in M[r][c]).
    
    # Let's compute the U, V, W parts for each (s, t).
    
    for s in range(7):
        for t in range(7):
            # Compute U part
            # For each A_block[i][j], find its coefficient.
            # A_block[i][j] is at row 2*i, col 2*j in the 4x4 matrix.
            # Its entries are at indices (2*i)*4 + 2*j + dr + 4*dc for dr,dc in {0,1}.
            
            # First, compute the coefficient for each A_block[i][j].
            # M is a 2x2 matrix of blocks. M[r][c] is a linear combination of A_blocks.
            # M_coeffs[s] gives the blocks in M.
            # Let's create a 2x2 matrix of coefficients for the A_blocks.
            M_block_coeffs = np.zeros((2,2,2,2))  # M_block_coeffs[r][c][i][j] = coeff of A_block[i][j] in M[r][c]
            for br, bc, coeff in M_coeffs[s]:
                M_block_coeffs[br][bc] = coeff  # Wait, M_coeffs[s] gives the blocks directly.
                # Actually, M_coeffs[s] is a list of (block_row, block_col, coeff) for the A_blocks that make up M.
                # So M[r][c] = sum of coeff * A_block[i][j] where (i,j) is in M_coeffs[s] with r=i, c=j?
                # No, M_coeffs[s] defines M as a 2x2 matrix of blocks.
                # e.g., s=0: M = [[A00, 0], [0, A11]].
                # So M[0][0] = A00, M[0][1] = 0, M[1][0] = 0, M[1][1] = A11.
                # So for each (br, bc, coeff) in M_coeffs[s], M[br][bc] += coeff * A_block[br][bc].
                # Wait, the block index is (br, bc).
                # So M_block_coeffs[br][bc][br][bc] = coeff.
                # Let's correct this.
                pass
            
            # Let's redefine M_block_coeffs properly.
            M_block_coeffs = np.zeros((2,2,2,2))
            for br, bc, coeff in M_coeffs[s]:
                M_block_coeffs[br][bc][br][bc] = coeff
            
            N_block_coeffs = np.zeros((2,2,2,2))
            for br, bc, coeff in N_coeffs[s]:
                N_block_coeffs[br][bc][br][bc] = coeff
            
            # Now, for the 2x2 Strassen factor t:
            # alpha_2[t] is a 2x2 matrix. alpha_2[t][r][c] multiplies M[r][c].
            # M[r][c] = sum_{i,j} M_block_coeffs[r][c][i][j] * A_block[i][j].
            # So the coefficient for A_block[i][j] is sum_{r,c} alpha_2[t][r][c] * M_block_coeffs[r][c][i][j].
            
            for i in range(2):
                for j in range(2):
                    coeff_A = 0.0
                    for r in range(2):
                        for c in range(2):
                            coeff_A += alpha_2[t][r][c] * M_block_coeffs[r][c][i][j]
                    if coeff_A != 0:
                        # A_block[i][j] entries are at (2*i + dr) * 4 + (2*j + dc)
                        for dr in range(2):
                            for dc in range(2):
                                u_idx = (2*i + dr) * 4 + (2*j + dc)
                                U[u_idx, col] += coeff_A
            
            # Similarly for V part
            for i in range(2):
                for j in range(2):
                    coeff_B = 0.0
                    for r in range(2):
                        for c in range(2):
                            coeff_B += beta_2[t][r][c] * N_block_coeffs[r][c][i][j]
                    if coeff_B != 0:
                        # B_block[i][j] entries are at (2*i + dr) * 4 + (2*j + dc)
                        # But B is 4x4, so indices are (2*i+dr)*4 + (2*j+dc)
                        for dr in range(2):
                            for dc in range(2):
                                v_idx = (2*i + dr) * 4 + (2*j + dc)
                                V[v_idx, col] += coeff_B
            
            # W part
            # recon_2[t] gives (out_r, out_c, gamma) for the 2x2 output matrix.
            # The output matrix is C_block[i][j] for i,j in {0,1}.
            # C_block[i][j] entries are at (2*i + dr) * 4 + (2*j + dc)
            for out_r, out_c, gamma in recon_2[t]:
                # The output block is C_block[out_r][out_c].
                # But wait, the 2x2 Strassen decomposition is for the product of two 2x2 matrices.
                # The output is a 2x2 matrix of blocks.
                # So the output block at (out_r, out_c) is C_block[out_r][out_c].
                # Its entries are at (2*out_r + dr) * 4 + (2*out_c + dc)
                for dr in range(2):
                    for dc in range(2):
                        w_idx = (2*out_r + dr) * 4 + (2*out_c + dc)
                        W[w_idx, col] += gamma
            
            col += 1
    
    # --- Part 2: 4x4 * 4x2 using block Strassen (rank 28) ---
    # B_right is the last 2 columns of B (4x2).
    # A is 4x4.
    # C_right is the last 2 columns of C (4x2).
    
    # Split A into A_top (2x4) and A_bot (2x4).
    # B_right = [B_top; B_bot] where B_top is 2x2, B_bot is 2x2.
    # C_top (2x2) = A_top * B_right.
    # A_top = [A_top_00, A_top_01] where A_top_00 = A[0:2, 0:2], A_top_01 = A[0:2, 2:4].
    # B_right = [B_top; B_bot] where B_top = B[0:2, 4:6], B_bot = B[2:4, 4:6].
    # C_top = A_top_00 * B_top + A_top_01 * B_bot.
    
    # This is sum of two 2x2*2x2 products.
    # Each has rank 7. Total rank 14.
    
    # Similarly for C_bot (2x2) = A_bot * B_right.
    # A_bot = [A_bot_00, A_bot_01] where A_bot_00 = A[2:4, 0:2], A_bot_01 = A[2:4, 2:4].
    # C_bot = A_bot_00 * B_top + A_bot_01 * B_bot.
    # Rank 14.
    
    # Total rank 28.
    
    # We will iterate over the two products for C_top and two for C_bot.
    # For each product, we use the 2x2 Strassen factors.
    
    # Product 1: A_top_00 * B_top
    # A_top_00 = A[0:2, 0:2] -> indices 0,1,4,5
    # B_top = B[0:2, 4:6] -> indices 4,5,10,11
    # C_top_00 = C[0:2, 4:6] -> indices 4,5,8,9 (W index: col 4,5; row 0,1 -> w_idx = col*4+row)
    # Wait, W index is (p*n, R) = (24, R). W index = j*4 + i.
    # C[0:2, 4:6] -> i=0,1; j=4,5.
    # w_idx = 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21.
    
    # Let's define a helper to add a 2x2*2x2 product.
    def add_2x2_product(U, V, W, col, A_indices, B_indices, C_indices):
        # A_indices: list of 4 indices for A[0:2, 0:2] in row-major order.
        # B_indices: list of 4 indices for B[0:2, 0:2] in row-major order.
        # C_indices: list of 4 indices for C[0:2, 0:2] in row-major order.
        for t in range(7):
            for a_idx in range(4):
                if alpha_2[t][a_idx//2][a_idx%2] != 0:
                    U[A_indices[a_idx], col] += alpha_2[t][a_idx//2][a_idx%2]
            for b_idx in range(4):
                if beta_2[t][b_idx//2][b_idx%2] != 0:
                    V[B_indices[b_idx], col] += beta_2[t][b_idx//2][b_idx%2]
            for out_r, out_c, gamma in recon_2[t]:
                C_idx = out_r * 2 + out_c
                W[C_indices[C_idx], col] += gamma
            col += 1
        return col
    
    # Product 1: A_top_00 * B_top
    A_top_00_indices = [0, 1, 4, 5]
    B_top_indices = [4, 5, 10, 11]
    C_top_00_indices = [16, 17, 20, 21]
    col = add_2x2_product(U, V, W, col, A_top_00_indices, B_top_indices, C_top_00_indices)
    
    # Product 2: A_top_01 * B_bot
    A_top_01_indices = [2, 3, 6, 7]
    B_bot_indices = [16, 17, 22, 23]
    C_top_01_indices = [18, 19, 22, 23]
    col = add_2x2_product(U, V, W, col, A_top_01_indices, B_bot_indices, C_top_01_indices)
    
    # Product 3: A_bot_00 * B_top
    A_bot_00_indices = [8, 9, 12, 13]
    B_top_indices = [4, 5, 10, 11]
    C_bot_00_indices = [20, 21, 24, 25] # Wait, C is 4x6. C[2:4, 4:6] -> i=2,3; j=4,5.
    # w_idx = 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23.
    # Let's recompute C indices.
    # C is 4x6. W shape is (24, R). W index = j*4 + i.
    # C[2:4, 4:6] -> i=2,3; j=4,5.
    # w_idx for (i=2, j=4) = 4*4+2 = 18.
    # w_idx for (i=3, j=4) = 4*4+3 = 19.
    # w_idx for (i=2, j=5) = 5*4+2 = 22.
    # w_idx for (i=3, j=5) = 5*4+3 = 23.
    # So C_bot_00_indices = [18, 19, 22, 23].
    # But we already used these for C_top_01?
    # C_top_01 is C[0:2, 4:6] -> i=0,1; j=4,5.
    # w_idx for (i=0, j=4) = 16.
    # w_idx for (i=1, j=4) = 17.
    # w_idx for (i=0, j=5) = 20.
    # w_idx for (i=1, j=5) = 21.
    # So C_top_01_indices = [16, 17, 20, 21].
    # And C_top_00_indices was for C[0:2, 4:6]? No, C_top_00 is C[0:2, 4:6] from A_top_00 * B_top.
    # Wait, C_top is 2x2. C_top[0:2, 0:2] is C[0:2, 4:6].
    # So C_top_00_indices = [16, 17, 20, 21].
    # C_top_01_indices = [18, 19, 22, 23]? No, C_top is 2x2.
    # C_top = [C_top_00, C_top_01].
    # C_top_00 is C[0:2, 4:5]? No, C_top is 2x2, so C_top_00 is C[0:2, 4:6]?
    # No, C_top is 2x2. C_top[0,0] = C[0,4], C_top[0,1] = C[0,5], C_top[1,0] = C[1,4], C_top[1,1] = C[1,5].
    # So C_top_00_indices = [16, 20, 17, 21]?
    # Let's be careful.
    # C_top is a 2x2 matrix.
    # C_top[0,0] = C[0,4] -> w_idx = 4*4+0 = 16.
    # C_top[0,1] = C[0,5] -> w_idx = 5*4+0 = 20.
    # C_top[1,0] = C[1,4] -> w_idx = 4*4+1 = 17.
    # C_top[1,1] = C[1,5] -> w_idx = 5*4+1 = 21.
    # So C_top_indices = [16, 20, 17, 21].
    
    # C_bot is a 2x2 matrix.
    # C_bot[0,0] = C[2,4] -> w_idx = 4*4+2 = 18.
    # C_bot[0,1] = C[2,5] -> w_idx = 5*4+2 = 22.
    # C_bot[1,0] = C[3,4] -> w_idx = 4*4+3 = 19.
    # C_bot[1,1] = C[3,5] -> w_idx = 5*4+3 = 23.
    # So C_bot_indices = [18, 22, 19, 23].
    
    # Let's redo the C indices.
    C_top_indices = [16, 20, 17, 21]
    C_bot_indices = [18, 22, 19, 23]
    
    # Product 1: A_top_00 * B_top -> C_top_00 (which is C_top)
    # Wait, A_top_00 * B_top is a 2x2 matrix.
    # A_top_00 = A[0:2, 0:2]. B_top = B[0:2, 4:6].
    # Product is C[0:2, 4:6].
    # So C_top_indices is correct.
    
    # Product 2: A_top_01 * B_bot -> C_top_01 (which is also C_top)
    # A_top_01 = A[0:2, 2:4]. B_bot = B[2:4, 4:6].
    # Product is C[0:2, 4:6].
    # So we add to the same C_top_indices.
    
    # Product 3: A_bot_00 * B_top -> C_bot_00 (which is C_bot)
    # A_bot_00 = A[2:4, 0:2]. B_top = B[0:2, 4:6].
    # Product is C[2:4, 4:6].
    # So C_bot_indices is correct.
    
    # Product 4: A_bot_01 * B_bot -> C_bot_01 (which is C_bot)
    # A_bot_01 = A[2:4, 2:4]. B_bot = B[2:4, 4:6].
    # Product is C[2:4, 4:6].
    # So we add to the same C_bot_indices.
    
    # Let's redo the add_2x2_product calls with correct C indices.
    # We already added Product 1 and 2 with wrong C indices.
    # Let's reset and redo.
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    
    # Part 1: 4x4 * 4x4 (rank 49)
    for s in range(7):
        for t in range(7):
            M_block_coeffs = np.zeros((2,2,2,2))
            for br, bc, coeff in M_coeffs[s]:
                M_block_coeffs[br][bc][br][bc] = coeff
            N_block_coeffs = np.zeros((2,2,2,2))
            for br, bc, coeff in N_coeffs[s]:
                N_block_coeffs[br][bc][br][bc] = coeff
            
            for i in range(2):
                for j in range(2):
                    coeff_A = 0.0
                    for r in range(2):
                        for c in range(2):
                            coeff_A += alpha_2[t][r][c] * M_block_coeffs[r][c][i][j]
                    if coeff_A != 0:
                        for dr in range(2):
                            for dc in range(2):
                                u_idx = (2*i + dr) * 4 + (2*j + dc)
                                U[u_idx, col] += coeff_A
            
            for i in range(2):
                for j in range(2):
                    coeff_B = 0.0
                    for r in range(2):
                        for c in range(2):
                            coeff_B += beta_2[t][r][c] * N_block_coeffs[r][c][i][j]
                    if coeff_B != 0:
                        for dr in range(2):
                            for dc in range(2):
                                v_idx = (2*i + dr) * 4 + (2*j + dc)
                                V[v_idx, col] += coeff_B
            
            for out_r, out_c, gamma in recon_2[t]:
                for dr in range(2):
                    for dc in range(2):
                        w_idx = (2*out_r + dr) * 4 + (2*out_c + dc)
                        W[w_idx, col] += gamma
            col += 1
    
    # Part 2: 4x4 * 4x2 (rank 28)
    # Product 1: A_top_00 * B_top
    A_top_00_indices = [0, 1, 4, 5]
    B_top_indices = [4, 5, 10, 11]
    C_top_indices = [16, 20, 17, 21]
    col = add_2x2_product(U, V, W, col, A_top_00_indices, B_top_indices, C_top_indices)
    
    # Product 2: A_top_01 * B_bot
    A_top_01_indices = [2, 3, 6, 7]
    B_bot_indices = [16, 17, 22, 23]
    col = add_2x2_product(U, V, W, col, A_top_01_indices, B_bot_indices, C_top_indices)
    
    # Product 3: A_bot_00 * B_top
    A_bot_00_indices = [8, 9, 12, 13]
    C_bot_indices = [18, 22, 19, 23]
    col = add_2x2_product(U, V, W, col, A_bot_00_indices, B_top_indices, C_bot_indices)
    
    # Product 4: A_bot_01 * B_bot
    A_bot_01_indices = [10, 11, 14, 15]
    col = add_2x2_product(U, V, W, col, A_bot_01_indices, B_bot_indices, C_bot_indices)
    
    # Validate
    if (time.time() - _t0) < 130.0:
        np.random.seed(12345)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 6)
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        C_computed = np.zeros((4, 6))
        for r in range(R):
            u_dot_A = np.dot(U_arr[:, r], A_flat)
            v_dot_B = np.dot(V_arr[:, r], B_flat)
            if u_dot_A != 0 and v_dot_B != 0:
                C_computed += (u_dot_A * v_dot_B) * W_arr[:, r].reshape(6, 4).T
        
        C_expected = A_test @ B_test
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook rank 96
            R_s = 96
            U_s = np.zeros((16, R_s))
            V_s = np.zeros((24, R_s))
            W_s = np.zeros((24, R_s))
            idx = 0
            for i in range(4):
                for k in range(4):
                    for j in range(6):
                        U_s[i * 4 + k, idx] = 1
                        V_s[k * 6 + j, idx] = 1
                        W_s[j * 4 + i, idx] = 1
                        idx += 1
            def to_list(mat):
                return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
            return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
