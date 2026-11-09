"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is 81 (49 for 4x4x4 + 32 for 4x4x2)
    rank_444 = 49
    rank_442 = 32
    total_rank = rank_444 + rank_442
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 4x4x4 using Block Strassen (Rank 49)
    # A is 4x4, B is 4x4, C is 4x4
    # Blocks are 2x2
    # We need to map the 2x2x2 Strassen tensors to the 4x4x4 indices
    
    # Strassen 2x2x2 tensors
    # U, V, W are 4x1 vectors for 2x2 matrices
    s_u = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [1, 0, 1, 0], # M6: A00, A10 (A10 - A00) -> [ -1, 0, 1, 0]? 
                      # M6 = (A10 - A00)(B00 + B01)
                      # U: A10=1, A00=-1. -> [ -1, 0, 1, 0]
        [0, 1, 0, -1] # M7: A01, A11 (A01 - A11) -> [ 0, 1, 0, -1]
    ]
    # Correction on M6, M7 signs based on formula:
    # M6 = (A10 - A00)(B00 + B01) -> U: [-1, 0, 1, 0]
    # M7 = (A01 - A11)(B10 + B11) -> U: [0, 1, 0, -1]
    
    s_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    s_v = [
        [1, 1, 0, 0], # M1: B00, B11 -> [1, 0, 0, 1]? 
                      # M1 = (A00+A11)(B00+B11). V: B00=1, B11=1. -> [1, 0, 0, 1]
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11 -> [0, 1, 0, -1]
        [-1, 0, 1, 0], # M4: B10 - B00 -> [-1, 0, 1, 0]
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01 -> [1, 1, 0, 0]
        [0, 0, 1, 1]  # M7: B10 + B11 -> [0, 0, 1, 1]
    ]
    # Correction on V:
    # M1: B00, B11 -> [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    # M3: B01 - B11 -> [0, 1, 0, -1]
    # M4: B10 - B00 -> [-1, 0, 1, 0]
    # M5: B11 -> [0, 0, 0, 1]
    # M6: B00 + B01 -> [1, 1, 0, 0]
    # M7: B10 + B11 -> [0, 0, 1, 1]
    
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    s_w = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, 0], # M2: C10 (index 1,0 -> 1)
        [0, 0, 1, 0], # M3: C01 (index 0,1 -> 1)
        [1, 1, 0, 0], # M4: C00, C10
        [0, 0, 1, 0], # M5: C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    # Correction on W:
    # M1: C00, C11 -> [1, 0, 0, 1]
    # M2: C10 -> [0, 1, 0, 0]
    # M3: C01 -> [0, 0, 1, 0]
    # M4: C00, C10 -> [1, 1, 0, 0]
    # M5: C01 -> [0, 0, 1, 0]
    # M6: C11 -> [0, 0, 0, 1]
    # M7: C00 -> [1, 0, 0, 0]
    
    s_w = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Map 2x2 indices to 4x4 indices
    # 4x4 matrix indices (i, j) where i, j in 0..3
    # Block (bi, bj) of size 2x2
    # i = 2*bi + di, j = 2*bj + dj
    
    # For each of the 4 blocks of A and B, we apply the 7 Strassen terms
    # Actually, the block Strassen algorithm computes 7 products of 2x2 blocks.
    # Each product is a 2x2x2 tensor.
    # So we have 7 sets of 7 rank-1 tensors? No.
    # The 7 products are $M_k = A_{block} B_{block}$.
    # Each $M_k$ is a 2x2 matrix.
    # The rank of $M_{<2,2,2>}$ is 7.
    # So each $M_k$ requires 7 rank-1 tensors.
    # Total 7 * 7 = 49 rank-1 tensors.
    
    # The blocks are:
    # A00, A01, A10, A11
    # B00, B01, B10, B11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We define the block coefficients for U, V, W
    # u_blocks: list of 7 lists of 4 blocks (coeff for A00, A01, A10, A11)
    # v_blocks: list of 7 lists of 4 blocks (coeff for B00, B01, B10, B11)
    # w_blocks: list of 7 lists of 4 blocks (coeff for C00, C01, C10, C11)
    
    u_blocks = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [-1, 0, 1, 0], # M6: A00, A10
        [0, 1, 0, -1]  # M7: A01, A11
    ]
    
    v_blocks = [
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01, B11
        [-1, 0, 1, 0], # M4: B00, B10
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 0, 1, 1]  # M7: B10, B11
    ]
    
    w_blocks = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, 0], # M2: C10
        [0, 1, 0, 0], # M3: C01
        [1, 0, 1, 0], # M4: C00, C10
        [0, 1, 0, 0], # M5: C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    col = 0
    for k in range(7):
        # For each Strassen product M_k, we use the 7 rank-1 tensors of 2x2x2
        for t in range(7):
            # u_blocks[k] gives the block coefficients for A
            # s_u[t] gives the internal 2x2 coefficients
            # We need to combine them
            
            # U vector
            for bi in range(2):
                for bj in range(2):
                    coeff_u = u_blocks[k][bi*2+bj] * s_u[t][bi*2+bj]
                    if coeff_u != 0:
                        # Map block (bi, bj) and internal (di, dj) to 4x4 index
                        # bi, bj are block indices (0,1)
                        # di, dj are internal indices (0,1)
                        # Wait, s_u[t] is a 4-vector [A00, A01, A10, A11] of the 2x2 block?
                        # No, s_u[t] is the vector for the 2x2 matrix in the 2x2x2 tensor.
                        # The 2x2 matrix corresponds to the block.
                        # So if u_blocks[k][block_idx] is non-zero, we scale the s_u[t] vector.
                        
                        # Actually, u_blocks[k] is a 4-vector [c00, c01, c10, c11] for the 4 blocks of A.
                        # s_u[t] is a 4-vector [u00, u01, u10, u11] for the 2x2 block.
                        # This interpretation is wrong.
                        
                        # Correct interpretation:
                        # M_k is a product of two 2x2 matrices.
                        # The 2x2 matrices are linear combinations of the 4 blocks of A and B.
                        # e.g. M1 = (A00+A11)(B00+B11).
                        # The "A-matrix" for M1 is A_M1 = A00 + A11.
                        # The "B-matrix" for M1 is B_M1 = B00 + B11.
                        # We decompose the product A_M1 B_M1 using Strassen 2x2x2.
                        # The Strassen decomposition gives 7 rank-1 terms.
                        # Each term is u_r (2x2) v_r (2x2) w_r (2x2).
                        # We need to expand u_r (2x2) into the 4x4 A.
                        # u_r is a 2x2 matrix. It corresponds to the structure of the block.
                        # But A_M1 is a sum of blocks.
                        # So u_r (2x2) is multiplied by the block coefficients?
                        # No.
                        # A_M1 = sum_{blocks} c_block * Block.
                        # The tensor for A_M1 B_M1 is (sum c_a A_a) (sum c_b B_b).
                        # This is not a single 2x2x2 tensor.
                        
                        # Wait. The block Strassen algorithm works like this:
                        # We compute 7 products of 2x2 matrices.
                        # Each product is a standard 2x2 matrix multiplication.
                        # So for each of the 7 products, we use the 7 rank-1 tensors of the 2x2x2 multiplication.
                        # The "A" in the 2x2x2 tensor is the 2x2 matrix formed by the linear combination of blocks.
                        # But the rank-1 tensors are defined on the 2x2 indices.
                        # How do we map them to the 4x4 indices?
                        
                        # Let's say for M1, A_M1 = A00 + A11.
                        # The 2x2 indices of A_M1 correspond to the 2x2 indices of the blocks.
                        # So A_M1[0,0] = A00[0,0] + A11[0,0].
                        # A_M1[0,1] = A00[0,1] + A11[0,1].
                        # etc.
                        
                        # So for a rank-1 term u_r v_r w_r of the 2x2x2 tensor:
                        # u_r is a 2x2 matrix.
                        # The contribution to the 4x4 U matrix is:
                        # For each block (bi, bj) with coefficient c:
                        #   For each internal index (di, dj):
                        #     U[2*bi+di, 2*bj+dj] += c * u_r[di, dj]
                        
                        # This works!
                        
                        pass # Placeholder for logic below

            # Logic for U
            for bi in range(2):
                for bj in range(2):
                    c = u_blocks[k][bi*2+bj]
                    if c == 0: continue
                    for di in range(2):
                        for dj in range(2):
                            val = c * s_u[t][di*2+dj]
                            if val != 0:
                                idx_u = (2*bi + di) * 4 + (2*bj + dj)
                                U[idx_u, col] += val
                                
            # Logic for V
            for bi in range(2):
                for bj in range(2):
                    c = v_blocks[k][bi*2+bj]
                    if c == 0: continue
                    for di in range(2):
                        for dj in range(2):
                            val = c * s_v[t][di*2+dj]
                            if val != 0:
                                idx_v = (2*bi + di) * 4 + (2*bj + dj)
                                V[idx_v, col] += val
                                
            # Logic for W
            for bi in range(2):
                for bj in range(2):
                    c = w_blocks[k][bi*2+bj]
                    if c == 0: continue
                    for di in range(2):
                        for dj in range(2):
                            val = c * s_w[t][di*2+dj]
                            if val != 0:
                                # W index is k*n + i. Here n=4.
                                # C is 4x4.
                                # C block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
                                # W index for C[2*bi+di, 2*bj+dj] is (2*bj+dj)*4 + (2*bi+di).
                                idx_w = (2*bj + dj) * 4 + (2*bi + di)
                                W[idx_w, col] += val
                                
            col += 1
            
    # Part 2: 4x4x2 using Schoolbook (Rank 32)
    # A is 4x4, B is 4x2, C is 4x2
    # B columns 0, 1.
    # C[0,0] = sum A[i,0..3] B[0..3,0]
    # This is 16 products.
    # Total 32 products.
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                # U: A[i, j]
                U[i * 4 + j, col] = 1
                # V: B[j, k]
                # B is 4x2. Index j*2 + k.
                V[j * 2 + k, col] = 1
                # W: C[i, k]
                # C is 4x2. Index k*4 + i.
                W[k * 4 + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
