import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to get index in U (A matrix)
    def u_idx(i, j):
        return i * m + j

    # Helper to get index in V (B matrix)
    def v_idx(j, k):
        return j * p + k

    # Helper to get index in W (C matrix)
    # Convention: W[k*n + i, r]
    def w_idx(k, i):
        return k * n + i

    # Define the 7 Strassen terms for 2x2 matrices
    # Each term is defined by linear combinations of the 4 blocks of A and B.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # We will iterate over the 7 block-level Strassen terms.
    # For each block-level term, we perform a 2x2 matrix multiplication using 7 rank-1 terms.
    
    # The 7 block-level combinations:
    # 1. M1 = (A00 + A11) * (B00 + B11)
    # 2. M2 = (A10 + A11) * B00
    # 3. M3 = A00 * (B01 - B11)
    # 4. M4 = A11 * (B10 - B00)
    # 5. M5 = (A00 + A01) * B11
    # 6. M6 = (A10 - A00) * (B00 + B01)
    # 7. M7 = (A01 - A11) * (B10 + B11)
    
    # For each block multiplication, we need to map the 2x2 blocks to the 4x4 indices.
    # Block (bi, bj) of A corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # Block (bi, bj) of B corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # Block (bi, bj) of C corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    
    # We define the 7 sub-terms for a 2x2 multiplication.
    # Let A_sub and B_sub be 2x2 matrices.
    # C_sub = A_sub * B_sub.
    # The 7 rank-1 terms for C_sub are:
    # t0: (A00+A11) * (B00+B11) -> C00 += t0, C11 += t0
    # t1: (A10+A11) * B00 -> C10 += t1, C11 -= t1
    # t2: A00 * (B01-B11) -> C01 += t2, C11 += t2
    # t3: A11 * (B10-B00) -> C00 += t3, C10 += t3
    # t4: (A00+A01) * B11 -> C00 -= t4, C01 += t4
    # t5: (A10-A00) * (B00+B01) -> C11 += t5
    # t6: (A01-A11) * (B10+B11) -> C00 += t6
    
    # Note: The indices in the sub-terms refer to the local 2x2 indices.
    
    # We will loop over the 7 block-level terms.
    # For each block-level term, we have specific linear combinations of the 4x4 blocks.
    # We need to map these to the 16 elements of A, B, C.
    
    # Let's define the block-level terms as lists of (block_coord, coeff) for A and B.
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    
    # Term 0: M1
    A_blocks_0 = [((0,0), 1), ((1,1), 1)]
    B_blocks_0 = [((0,0), 1), ((1,1), 1)]
    C_contrib_0 = [((0,0), 1), ((1,1), 1)]
    
    # Term 1: M2
    A_blocks_1 = [((1,0), 1), ((1,1), 1)]
    B_blocks_1 = [((0,0), 1)]
    C_contrib_1 = [((1,0), 1), ((1,1), -1)]
    
    # Term 2: M3
    A_blocks_2 = [((0,0), 1)]
    B_blocks_2 = [((0,1), 1), ((1,1), -1)]
    C_contrib_2 = [((0,1), 1), ((1,1), 1)]
    
    # Term 3: M4
    A_blocks_3 = [((1,1), 1)]
    B_blocks_3 = [((1,0), 1), ((0,0), -1)]
    C_contrib_3 = [((0,0), 1), ((1,0), 1)]
    
    # Term 4: M5
    A_blocks_4 = [((0,0), 1), ((0,1), 1)]
    B_blocks_4 = [((1,1), 1)]
    C_contrib_4 = [((0,0), -1), ((0,1), 1)]
    
    # Term 5: M6
    A_blocks_5 = [((1,0), 1), ((0,0), -1)]
    B_blocks_5 = [((0,0), 1), ((0,1), 1)]
    C_contrib_5 = [((1,1), 1)]
    
    # Term 6: M7
    A_blocks_6 = [((0,1), 1), ((1,1), -1)]
    B_blocks_6 = [((1,0), 1), ((1,1), 1)]
    C_contrib_6 = [((0,0), 1)]
    
    block_terms = [
        (A_blocks_0, B_blocks_0, C_contrib_0),
        (A_blocks_1, B_blocks_1, C_contrib_1),
        (A_blocks_2, B_blocks_2, C_contrib_2),
        (A_blocks_3, B_blocks_3, C_contrib_3),
        (A_blocks_4, B_blocks_4, C_contrib_4),
        (A_blocks_5, B_blocks_5, C_contrib_5),
        (A_blocks_6, B_blocks_6, C_contrib_6)
    ]
    
    # For each block term, we expand it into 7 rank-1 terms for the 2x2 multiplication.
    # The 2x2 multiplication terms (local indices):
    # t0: A(0,0)+A(1,1) * B(0,0)+B(1,1)
    # t1: A(1,0)+A(1,1) * B(0,0)
    # t2: A(0,0) * B(0,1)-B(1,1)
    # t3: A(1,1) * B(1,0)-B(0,0)
    # t4: A(0,0)+A(0,1) * B(1,1)
    # t5: A(1,0)-A(0,0) * B(0,0)+B(0,1)
    # t6: A(0,1)-A(1,1) * B(1,0)+B(1,1)
    
    # We need to map the local 2x2 indices to the global 4x4 indices.
    # Local (r, c) in a block at (bi, bj) maps to global (2*bi+r, 2*bj+c).
    
    # For a given block term (A_comb, B_comb, C_comb):
    # A_comb is a list of ((bi, bj), coeff).
    # We need to form the linear combination of A elements.
    # For local index (lr, lc), the global element is at (2*bi+lr, 2*bj+lc).
    # So for a term like A(0,0) in the local 2x2, it corresponds to A elements at (2*bi+0, 2*bj+0) for all (bi,bj) in A_comb with their coefficients.
    
    rank_idx = 0
    for A_comb, B_comb, C_comb in block_terms:
        for k in range(7):
            if time.time() - start_time > DEADLINE:
                return {"U": U, "V": V, "W": W}
            
            # Define the local 2x2 indices for the k-th term
            # Returns (A_local_indices, B_local_indices, C_local_indices)
            # Each is a list of ((r,c), coeff)
            
            if k == 0:
                A_loc = [((0,0), 1), ((1,1), 1)]
                B_loc = [((0,0), 1), ((1,1), 1)]
                C_loc = [((0,0), 1), ((1,1), 1)]
            elif k == 1:
                A_loc = [((1,0), 1), ((1,1), 1)]
                B_loc = [((0,0), 1)]
                C_loc = [((1,0), 1), ((1,1), -1)]
            elif k == 2:
                A_loc = [((0,0), 1)]
                B_loc = [((0,1), 1), ((1,1), -1)]
                C_loc = [((0,1), 1), ((1,1), 1)]
            elif k == 3:
                A_loc = [((1,1), 1)]
                B_loc = [((1,0), 1), ((0,0), -1)]
                C_loc = [((0,0), 1), ((1,0), 1)]
            elif k == 4:
                A_loc = [((0,0), 1), ((0,1), 1)]
                B_loc = [((1,1), 1)]
                C_loc = [((0,0), -1), ((0,1), 1)]
            elif k == 5:
                A_loc = [((1,0), 1), ((0,0), -1)]
                B_loc = [((0,0), 1), ((0,1), 1)]
                C_loc = [((1,1), 1)]
            else: # k == 6
                A_loc = [((0,1), 1), ((1,1), -1)]
                B_loc = [((1,0), 1), ((1,1), 1)]
                C_loc = [((0,0), 1)]
                
            # Now combine A_comb and A_loc to get global A elements
            # U elements
            U_elems = {}
            for (bi, bj), coeff_block in A_comb:
                for (lr, lc), coeff_loc in A_loc:
                    gi = 2 * bi + lr
                    gj = 2 * bj + lc
                    idx = u_idx(gi, gj)
                    U_elems[idx] = U_elems.get(idx, 0) + coeff_block * coeff_loc
            
            # V elements
            V_elems = {}
            for (bi, bj), coeff_block in B_comb:
                for (lr, lc), coeff_loc in B_loc:
                    gj = 2 * bi + lr
                    gk = 2 * bj + lc
                    idx = v_idx(gj, gk)
                    V_elems[idx] = V_elems.get(idx, 0) + coeff_block * coeff_loc
            
            # W elements
            # C_comb gives the block contributions.
            # C_loc gives the local contributions within that block.
            W_elems = {}
            for (ci, cj), coeff_block in C_comb:
                for (lr, lc), coeff_loc in C_loc:
                    gi = 2 * ci + lr
                    gj = 2 * cj + lc
                    # W index is k*n + i, where k is col, i is row
                    # Here gj is col, gi is row
                    idx = w_idx(gj, gi)
                    W_elems[idx] = W_elems.get(idx, 0) + coeff_block * coeff_loc
            
            # Update U, V, W
            for idx, val in U_elems.items():
                U[idx][rank_idx] += val
            for idx, val in V_elems.items():
                V[idx][rank_idx] += val
            for idx, val in W_elems.items():
                W[idx][rank_idx] += val
                
            rank_idx += 1

    return {"U": U, "V": V, "W": W}
