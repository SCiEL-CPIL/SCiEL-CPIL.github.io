import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = DIMENSIONS

    # Precompute top-level 3x3 block decomposition (rank 9)
    top_a = {}
    top_b = {}
    top_c = {}
    # 9 elementary 3x3 block products: a = A[i,j], b = B[j,k], c = C[i,k]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                top_a[(i, j)] = (i, j)
                top_b[(j, k)] = (j, k)
                top_c[(i, k)] = (i, k)

    # Strassen 2x2 decomposition (rank 7)
    # M_s = sum_r alpha_s(r) * A_block(r) * B_block(r)
    # C_block = sum_s beta_s * M_s
    # alpha_s is a list of (row, col, coeff) for A block
    # beta_s is a list of (row, col, coeff) for C block
    strassen = [
        # (A_rows_cols, B_rows_cols, C_rows_cols)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]

    # For 3x3 block multiplication using Strassen on 2x2 + trivial 1x1:
    # Split 3x3 into: top-left 2x2, top-right 2x1, bottom-left 1x2, bottom-right 1x1
    # Actually, let's use a different approach: decompose 3x3 as 4 sub-blocks
    # A_3x3 = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # We'll use a 3x3 -> 2x2 Strassen + 1x1 approach by grouping:
    # Group rows 0,1 together and row 2 separately (and same for cols)
    # This gives 4 groups: (0,1)x(0,1), (0,1)x(2), (2)x(0,1), (2)x(2)
    # For (0,1)x(0,1): use Strassen (rank 7)
    # For (0,1)x(2): this is 2x1, can use rank 2 (or 1 with sum)
    # For (2)x(0,1): this is 1x2, can use rank 2 (or 1 with sum)
    # For (2)x(2): rank 1
    # Total: 7 + 2 + 2 + 1 = 12? No, let me think more carefully.
    
    # Actually, let me use a cleaner approach: decompose 3x3 multiplication
    # by treating it as a block matrix and using Strassen-like formulas.
    # 
    # A 3x3 matrix can be written as:
    # [X  y]  where X is 2x2, y is 2x1
    # [z  w]  where z is 1x2, w is 1x1
    #
    # B 3x3 matrix:
    # [P  q]  where P is 2x2, q is 2x1
    # [r  s]  where r is 1x2, s is 1x1
    #
    # C = A*B:
    # Top-left 2x2: X*P (use Strassen, rank 7)
    # Top-right 2x1: X*q + y*s (need to compute)
    # Bottom-left 1x2: z*P + w*r
    # Bottom-right 1x1: z*q + w*s
    #
    # For top-right 2x1 = X*q + y*s:
    #   X*q is 2x1, y*s is 2x1
    #   We can compute X*q using 2 multiplications (each row of X times q)
    #   y*s uses 2 multiplications
    #   Total: 4 mults
    #
    # For bottom-left 1x2 = z*P + w*r:
    #   z*P is 1x2, w*r is 1x2
    #   z*P uses 2 multiplications
    #   w*r uses 2 multiplications
    #   Total: 4 mults
    #
    # For bottom-right 1x1 = z*q + w*s:
    #   z*q is 1x1 (inner product of two 1x2 vectors) = 2 mults
    #   w*s is 1x1 = 1 mult
    #   Total: 3 mults
    #
    # Grand total for 3x3: 7 + 4 + 4 + 3 = 18 mults
    # That's worse than 9! Let me reconsider.
    
    # Better approach: just use schoolbook 3x3 (rank 9) since Strassen doesn't help
    # for 3x3 directly. But we can still use the hierarchical approach where
    # the top level is 6x6 -> 3x3 blocks (rank 9), and each 3x3 is done with
    # schoolbook (rank 9), giving 9*9 = 81. That's worse than 189!
    
    # Wait, I'm confusing myself. Let me reconsider.
    # 
    # Current program: 6x6 -> 3x3 blocks, then each 3x3 block multiplication
    # is done using Strassen 2x2 on the 3x3 (treating it as 2x2 + 1x1 blocks).
    # That gives 9 * 7 = 63? No, the current program does 9 * 7 = 63 for the
    # block products, but each "block product" is a 2x2 multiplication of 3x3 blocks.
    # 
    # Actually, I think the current approach is:
    # - Decompose 6x6 into 3x3 blocks (3x3 grid)
    # - For each of the 9 block positions, compute the 3x3 block product
    # - Each 3x3 block product uses Strassen-like decomposition with rank 7
    # - Total rank: 9 * 7 = 63? But the code says 189...
    # 
    # Looking at the code more carefully: it uses block_size=2, num_blocks=3,
    # so it's actually decomposing 6x6 into 2x2 blocks (3x3 grid of 2x2 blocks).
    # Then each 2x2 block multiplication uses Strassen (rank 7).
    # Total rank: 9 * 7 = 63? But the code initializes rank=189...
    # 
    # Ah, I see - the code is wrong! It should be 9*7=63, not 189.
    # But the score is 1.1429 = 216/189, so rank=189 is being used.
    # 
    # Wait, 216/189 = 8/7 ≈ 1.1429. So the base is 216 (schoolbook 6x6)
    # and the improvement is to rank 189.
    # 
    # Let me just implement a clean hierarchical approach:
    # 6x6 -> 3x3 blocks (3x3 grid), each 3x3 block product uses rank 9 (schoolbook)
    # Total: 9 * 9 = 81. That's a huge improvement!
    # 
    # But wait, that's not right either. The 9 block products each need to
    # multiply two 3x3 matrices, which takes rank 9 each, so total rank = 9*9 = 81.
    # Score would be 216/81 = 8/3 ≈ 2.667.
    
    # Let me implement the simple approach: 6x6 -> 3x3 blocks, schoolbook within
    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # For each of the 9 block multiplications
    for i in range(3):
        for k in range(3):
            for j in range(3):
                # C[i,k] += A[i,j] * B[j,k]
                # A[i,j] is 3x3, B[j,k] is 3x3, C[i,k] is 3x3
                for ai in range(3):
                    for aj in range(3):
                        for bk in range(3):
                            # U row: (i*3+ai, j*3+aj)
                            u_row = (i * 3 + ai) * m + (j * 3 + aj)
                            # V row: (j*3+aj, k*3+bk)
                            v_row = (j * 3 + aj) * p + (k * 3 + bk)
                            # W row: (k*3+bk, i*3+ai) [trace convention W[k*n+i, r]]
                            w_row = (k * 3 + bk) * n + (i * 3 + ai)
                            U[u_row][col] = 1
                            V[v_row][col] = 1
                            W[w_row][col] = 1
                            col += 1
    
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
