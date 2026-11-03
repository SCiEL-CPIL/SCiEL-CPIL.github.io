"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START

def _strassen_2x2_factors():
    """
    Returns (U, V, W) for 2x2 x 2x2 matrix multiplication.
    U shape: (4, 7)  [i*2 + j, c]
    V shape: (4, 7)  [j*2 + k, c]
    W shape: (4, 7)  [k*2 + i, c]
    Rank 7.
    """
    n = 2
    rank = 7
    U = np.zeros((n * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * n, rank), dtype=float)

    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    # A00 is index 0, A11 is index 3
    # B00 is index 0, B11 is index 3
    # C00 is index 0, C11 is index 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10 + A11)B00 -> C10, C11
    # A10 is index 2, A11 is index 3
    # B00 is index 0
    # C10 is index 2, C11 is index 3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1

    # M3 = A00(B01 - B11) -> C01, C11
    # A00 is index 0
    # B01 is index 1, B11 is index 3
    # C01 is index 1, C11 is index 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10 - B00) -> C00, C10
    # A11 is index 3
    # B10 is index 2, B00 is index 0
    # C00 is index 0, C10 is index 2
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # M5 = (A00 + A01)B11 -> C00, C01
    # A00 is index 0, A01 is index 1
    # B11 is index 3
    # C00 is index 0, C01 is index 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1

    # M6 = (A10 - A00)(B00 + B01) -> C11
    # A10 is index 2, A00 is index 0
    # B00 is index 0, B01 is index 1
    # C11 is index 3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01 - A11)(B10 + B11) -> C00, C11
    # A01 is index 1, A11 is index 3
    # B10 is index 2, B11 is index 3
    # C00 is index 0, C11 is index 3
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1

    return U, V, W


def _strassen_4x4_factors():
    """
    Returns (U, V, W) for 4x4 x 4x4 matrix multiplication.
    U shape: (16, 49)
    V shape: (16, 49)
    W shape: (16, 49)
    Rank 49.
    Uses 2x2 blocks and the rank-7 2x2 factors.
    """
    n = 4
    m = 4
    p = 4
    U2, V2, W2 = _strassen_2x2_factors()
    rank2 = U2.shape[1] # 7
    rank_total = 7 * rank2 # 49
    
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)

    # Block indices for 4x4 into 2x2 blocks:
    # Block (b_i, b_j) where b_i, b_j in {0, 1}
    # Elements in block (b_i, b_j) of a 4x4 matrix X (flattened i*m+j) are:
    # (b_i*2 + r)*m + (b_j*2 + c) for r,c in {0,1}
    
    # We need to map the 2x2 factor indices to 4x4 factor indices.
    # For A (U):
    # U2 index u2 = i2*m2 + j2 where i2,j2 in {0,1} are block coords relative to the 2x2 subproblem.
    # The 2x2 subproblem operates on the 2x2 blocks of the 4x4.
    # So the 'row' of U2 corresponds to a pair of elements in A? No.
    # The 2x2 decomposition assumes A is 2x2. Here A is 4x4, viewed as 2x2 of 2x2 blocks.
    # The Strassen algorithm computes M = A_block * B_block.
    # The factors U2, V2, W2 are for the *block* level? No, they are for the *element* level of a 2x2 matrix.
    # To use them for 4x4, we treat the 4x4 multiplication as 7 multiplications of 2x2 matrices.
    # Each 2x2 matrix multiplication uses the rank-7 factors.
    # So for each of the 7 Strassen steps (let's call them s=0..6), we use the U2, V2, W2 factors.
    # The input to step s is linear combinations of the 2x2 blocks.
    # Let A be composed of blocks A00, A01, A10, A11.
    # Let B be composed of blocks B00, B01, B10, B11.
    # M_s = L_A_s(A_blocks) * L_B_s(B_blocks).
    # The factorization for M_s is given by U2, V2, W2.
    # U2 has shape (4, 7). The rows of U2 correspond to the elements of the 2x2 matrix being multiplied on the left.
    # The elements of the 2x2 matrix L_A_s are linear combinations of the elements of A_blocks.
    # Specifically, the (r, c) element of the left matrix in step s is a linear combination of A elements.
    # Let's determine the linear combinations for each step.
    
    # The 2x2 Strassen formulas operate on the blocks.
    # Step 0: M0 = (A00 + A11)(B00 + B11)
    # The left matrix is A00 + A11. Its elements are:
    # L00 = A00_00 + A11_00
    # L01 = A00_01 + A11_01
    # L10 = A00_10 + A11_10
    # L11 = A00_11 + A11_11
    # In the flattened 4x4 A (index i*4+j):
    # A00_00 is A[0,0] (idx 0)
    # A00_01 is A[0,1] (idx 1)
    # A00_10 is A[1,0] (idx 4)
    # A00_11 is A[1,1] (idx 5)
    # A11_00 is A[2,2] (idx 10)
    # A11_01 is A[2,3] (idx 11)
    # A11_10 is A[3,2] (idx 14)
    # A11_11 is A[3,3] (idx 15)
    
    # The U factor for step 0, column c (c in 0..6), will have non-zero entries at the A indices corresponding to L's elements.
    # U2[r, c] gives the coefficient for the r-th element of the left 2x2 matrix.
    # The r-th element of the left 2x2 matrix corresponds to a specific pair of A indices.
    # Let's map the 2x2 element index r (0..3) to the 4x4 indices it sums.
    # r=0 (top-left): A[0,0] and A[2,2]
    # r=1 (top-right): A[0,1] and A[2,3]
    # r=2 (bottom-left): A[1,0] and A[3,2]
    # r=3 (bottom-right): A[1,1] and A[3,3]
    
    # Generally, for a step with left matrix L = sum_alpha alpha_a * A_alpha_block,
    # the element L_r (r in 0..3) is sum_alpha alpha_a * A_alpha_block[local_r].
    # local_r maps to global indices.
    
    # Let's define the block structure.
    # Blocks are indexed by (bi, bj) in {0,1}^2.
    # Global index of A element in block (bi, bj) at local pos (lr, lc) is (bi*2+lr)*4 + (bj*2+lc).
    
    # We will precompute for each step s (0..6) and each local row r (0..3) and col c (0..3) of the 2x2 block,
    # the list of (global_index, coefficient) for U, V, W.
    
    # Strassen Steps Definitions:
    # s=0: L = A00 + A11, R = B00 + B11
    # s=1: L = A10 + A11, R = B00
    # s=2: L = A00, R = B01 - B11
    # s=3: L = A11, R = B10 - B00
    # s=4: L = A00 + A01, R = B11
    # s=5: L = A10 - A00, R = B00 + B01
    # s=6: L = A01 - A11, R = B10 + B11
    
    # C Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Note: The indices in C reconstruction refer to the block positions.
    # C00 is block (0,0), C01 is (0,1), C10 is (1,0), C11 is (1,1).
    
    # For each step s, we use U2[:, c], V2[:, c], W2[:, c] for c in 0..6.
    # The global column in U, V, W is s * 7 + c.
    
    # Let's build the mapping.
    
    # A Blocks:
    # A00: bi=0, bj=0
    # A01: bi=0, bj=1
    # A10: bi=1, bj=0
    # A11: bi=1, bj=1
    
    # For each step s, define the linear combination of blocks for Left (U) and Right (V).
    # U combines A blocks. V combines B blocks.
    # The local 2x2 structure of the combination is preserved.
    # E.g., if L = A00 + A11, then L[0,0] = A00[0,0] + A11[0,0].
    
    # We can represent the coefficients for each local 2x2 position (lr, lc) for U and V.
    # coeff_U[s][lr][lc] is a list of (block_idx, coeff) where block_idx is (bi, bj).
    # Actually, simpler: for each s, and each local position p in 0..3 (flattened lr*2+lc),
    # we have a vector of coefficients for the 4 blocks.
    
    # Let's define the block indices as 0:A00, 1:A01, 2:A10, 3:A11.
    
    # Step 0:
    # U: A00(1) + A11(1) for all positions.
    # V: B00(1) + B11(1) for all positions.
    # Step 1:
    # U: A10(1) + A11(1)
    # V: B00(1)
    # Step 2:
    # U: A00(1)
    # V: B01(1) - B11(1)
    # Step 3:
    # U: A11(1)
    # V: B10(1) - B00(1)
    # Step 4:
    # U: A00(1) + A01(1)
    # V: B11(1)
    # Step 5:
    # U: A10(1) - A00(1)
    # V: B00(1) + B01(1)
    # Step 6:
    # U: A01(1) - A11(1)
    # V: B10(1) + B11(1)
    
    # We need to map block index and local position to global index.
    # A block (bi, bj) local (lr, lc) -> global (bi*2+lr)*4 + (bj*2+lc)
    # B block (bi, bj) local (lr, lc) -> global (bi*2+lr)*4 + (bj*2+lc)  (since B is 4x4, j*4+k)
    # Wait, B is 4x4. j is 0..3, k is 0..3.
    # Block (bi, bj) of B corresponds to rows bi*2..bi*2+1, cols bj*2..bj*2+1.
    # So B element at global (j, k) where j=bi*2+lr, k=bj*2+lc.
    # Index is j*4 + k.
    
    # C block (ci, cj) local (lr, lc) -> global (ci*2+lr)*4 + (cj*2+lc)
    # W index is k*n + i. k is col of C, i is row of C.
    # C element at (i, k). i = ci*2+lr, k = cj*2+lc.
    # Index is k*4 + i.
    
    # Let's iterate over steps s=0..6.
    for s in range(7):
        # Determine block coefficients for U (A) and V (B)
        # u_blocks: dict from block_idx to coeff. But it's uniform across local positions.
        if s == 0:
            u_blocks = {0: 1, 3: 1}
            v_blocks = {0: 1, 3: 1}
        elif s == 1:
            u_blocks = {2: 1, 3: 1}
            v_blocks = {0: 1}
        elif s == 2:
            u_blocks = {0: 1}
            v_blocks = {1: 1, 3: -1}
        elif s == 3:
            u_blocks = {3: 1}
            v_blocks = {2: 1, 0: -1}
        elif s == 4:
            u_blocks = {0: 1, 1: 1}
            v_blocks = {3: 1}
        elif s == 5:
            u_blocks = {2: 1, 0: -1}
            v_blocks = {0: 1, 1: 1}
        elif s == 6:
            u_blocks = {1: 1, 3: -1}
            v_blocks = {2: 1, 3: 1}
        else:
            continue

        # For each local position p (0..3) in the 2x2 matrix, and each rank column c (0..6)
        for p in range(4):
            lr = p // 2
            lc = p % 2
            
            # U factor
            # The element of the left matrix at local (lr, lc) is sum of coeff * A_block(lr, lc)
            # The U2 factor for this local position is U2[p, c].
            # So we add U2[p, c] * coeff to U[global_index, s*7+c]
            for block_idx, coeff in u_blocks.items():
                bi = block_idx // 2
                bj = block_idx % 2
                global_i = bi * 2 + lr
                global_j = bj * 2 + lc
                global_idx = global_i * 4 + global_j
                U[global_idx, s * 7 + c] += U2[p, c] * coeff
                
            # V factor
            # The element of the right matrix at local (lr, lc) is sum of coeff * B_block(lr, lc)
            # The V2 factor for this local position is V2[p, c].
            for block_idx, coeff in v_blocks.items():
                bi = block_idx // 2
                bj = block_idx % 2
                global_j = bi * 2 + lr  # B row
                global_k = bj * 2 + lc  # B col
                global_idx = global_j * 4 + global_k
                V[global_idx, s * 7 + c] += V2[p, c] * coeff
                
            # W factor
            # The output C block is determined by s and the C reconstruction formula.
            # The local position (lr, lc) in the C block corresponds to global (i, k).
            # The W2 factor for this local position is W2[p, c].
            # But W2 is for the 2x2 output. The 2x2 output of step s contributes to specific C blocks.
            # We need to know which C blocks get the contribution from step s, and with what sign.
            
            # C Reconstruction:
            # C00 = M0 + M3 - M4 + M6
            # C01 = M2 + M4
            # C10 = M1 + M3
            # C11 = M0 - M1 + M2 + M5
            
            # Let's define c_blocks: dict from c_block_idx to coeff.
            c_blocks = {}
            if s == 0:
                c_blocks = {0: 1, 3: 1} # C00, C11
            elif s == 1:
                c_blocks = {2: 1, 3: 1} # C10, C11
            elif s == 2:
                c_blocks = {1: 1, 3: 1} # C01, C11
            elif s == 3:
                c_blocks = {0: 1, 2: 1} # C00, C10
            elif s == 4:
                c_blocks = {0: -1, 1: 1} # C00, C01
            elif s == 5:
                c_blocks = {3: 1} # C11
            elif s == 6:
                c_blocks = {0: 1, 3: -1} # C00, C11
            
            for block_idx, coeff in c_blocks.items():
                ci = block_idx // 2
                cj = block_idx % 2
                global_i = ci * 2 + lr
                global_k = cj * 2 + lc
                global_idx = global_k * 4 + global_i
                W[global_idx, s * 7 + c] += W2[p, c] * coeff

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    # We want to multiply 4x4 by 4x8.
    # This is equivalent to multiplying 4x4 by [B1 B2] where B1, B2 are 4x4.
    # Result C = [C1 C2] where C1 = A*B1, C2 = A*B2.
    # We can use the rank-49 decomposition for 4x4*4x4 for both B1 and B2.
    # Total rank = 49 + 49 = 98.
    
    U_base, V_base, W_base = _strassen_4x4_factors()
    
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # First half: Multiply A by B1 (cols 0-3 of B)
    # U part is same for both.
    U[:, 0:49] = U_base
    
    # V part: B1 is rows 0-3 of B (since B is 4x8, rows 0-3, cols 0-3)
    # V_base has shape (16, 49) for 4x4 * 4x4.
    # The rows of V correspond to j*p + k. For 4x4*4x4, j,k in 0..3, so index j*4+k.
    # In our 4x4*4x8, j in 0..3, k in 0..7.
    # For B1, k is 0..3. So the indices are j*8 + k.
    # We need to map V_base indices (j*4+k) to V indices (j*8+k).
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V_base[j * 4 + k, :]
            
    # W part: C1 is cols 0-3 of C (since C is 4x8, rows 0-3, cols 0-3)
    # W_base has shape (16, 49) for 4x4*4x4.
    # The rows of W correspond to k*n + i. For 4x4*4x4, k,i in 0..3, so index k*4+i.
    # In our 4x4*4x8, k in 0..7, i in 0..3.
    # For C1, k is 0..3. So the indices are k*4 + i.
    # This matches W_base indices exactly.
    W[:, 0:49] = W_base
    
    # Second half: Multiply A by B2 (cols 4-7 of B)
    U[:, 49:98] = U_base
    
    # V part: B2 is rows 0-3 of B, cols 4-7.
    # k is 4..7. Index is j*8 + k.
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V_base[j * 4 + k, :]
            
    # W part: C2 is cols 4-7 of C.
    # k is 4..7. Index is k*4 + i.
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, 49:98] = W_base[k * 4 + i, :]
            
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
