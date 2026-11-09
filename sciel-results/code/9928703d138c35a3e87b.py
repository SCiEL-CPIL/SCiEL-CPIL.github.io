"""Matrix multiplication 6x6x6 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # We will use a structured decomposition based on Strassen's algorithm
    # applied to 3x3 blocks, with schoolbook multiplication for the 3x3 blocks.
    # This gives a rank of 7 * 27 = 189.
    
    # Strassen's algorithm for 2x2 blocks of 3x3 matrices:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a 3x3 matrix product.
    # We use schoolbook for 3x3, which has rank 27.
    # Total rank = 7 * 27 = 189.
    
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a schoolbook decomposition for a 3x3 block
    # at specific row/col offsets in A, B, and C.
    # A block is at (row_start, col_start)
    # B block is at (row_start, col_start)
    # C block is at (row_start, col_start)
    
    def add_schoolbook_3x3(A_row, A_col, B_row, B_col, C_row, C_col, coeff_A, coeff_B, coeff_C, base_idx):
        # coeff_A, coeff_B, coeff_C are multipliers for the linear combinations
        # This function is a bit complex because Strassen uses sums/diffs.
        # Instead, we build the full U, V, W matrices by iterating over the 7 Strassen terms.
        pass

    # We will construct the matrices directly.
    # The indices for U are (i, j) where i in [0,6), j in [0,6).
    # The indices for V are (j, k) where j in [0,6), k in [0,6).
    # The indices for W are (k, i) where k in [0,6), i in [0,6).
    
    # Strassen terms:
    # M1: (A00+A11)(B00+B11)
    # A00 is rows 0-2, cols 0-2. A11 is rows 3-5, cols 3-5.
    # B00 is rows 0-2, cols 0-2. B11 is rows 3-5, cols 3-5.
    # Result C00 (rows 0-2, cols 0-2) and C11 (rows 3-5, cols 3-5) etc.
    
    # We define the 7 terms.
    # For each term, we define the linear combination of A, B, and the contribution to C.
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    # U_part: A00 + A11.
    # V_part: B00 + B11.
    # W_part: Contributes to C00, C01, C10, C11 with coefficients.
    # C00 += M1, C01 += M1 (No, C01 = M3+M5), C10 += M1 (No), C11 += M1.
    # Actually:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C00 (coeff 1) and C11 (coeff 1).
    # M2 contributes to C10 (coeff 1) and C11 (coeff -1).
    # M3 contributes to C00 (coeff 0? No, C00=M1+M4-M5+M7), C01 (coeff 1), C11 (coeff 1).
    # Wait, C00 = M1 + M4 - M5 + M7.
    # C01 = M3 + M5.
    # C10 = M2 + M4.
    # C11 = M1 - M2 + M3 + M6.
    
    # Let's list contributions:
    # M1: C00 (+1), C11 (+1)
    # M2: C10 (+1), C11 (-1)
    # M3: C01 (+1), C11 (+1)
    # M4: C00 (+1), C10 (+1)
    # M5: C00 (-1), C01 (+1)
    # M6: C11 (+1)
    # M7: C00 (+1)
    
    # Now we map the blocks to indices.
    # A00: i in [0,3), j in [0,3)
    # A01: i in [0,3), j in [3,6)
    # A10: i in [3,6), j in [0,3)
    # A11: i in [3,6), j in [3,6)
    
    # Same for B.
    
    # C00: i in [0,3), k in [0,3) -> W index k*6 + i
    # C01: i in [0,3), k in [3,6)
    # C10: i in [3,6), k in [0,3)
    # C11: i in [3,6), k in [3,6)
    
    # We will iterate over the 7 Strassen terms.
    # For each term, we define the A-blocks, B-blocks involved, and their coefficients.
    # Then we add the schoolbook decomposition for the 3x3 product, scaled by the coefficients.
    
    def add_term(A_blocks, B_blocks, C_contribs, base_idx):
        # A_blocks: list of (block_id, coeff)
        # B_blocks: list of (block_id, coeff)
        # C_contribs: list of (block_id, coeff)
        
        # The term is sum_A (coeff_A * A_block) * sum_B (coeff_B * B_block)
        # = sum_{a,b} coeff_A_a coeff_B_b A_a B_b
        # This is a sum of 3x3 matrix products.
        # We can decompose each A_a B_b using schoolbook.
        
        for a_id, a_coeff in A_blocks:
            for b_id, b_coeff in B_blocks:
                coeff = a_coeff * b_coeff
                if coeff == 0:
                    continue
                
                # Get indices for A_block a_id
                a_i_start, a_i_end, a_j_start, a_j_end = get_block_indices(a_id, 'A')
                # Get indices for B_block b_id
                b_j_start, b_j_end, b_k_start, b_k_end = get_block_indices(b_id, 'B')
                
                # Note: A_block (i, j), B_block (j, k).
                # The j indices must match.
                # In Strassen, the blocks are aligned.
                
                # Add schoolbook for this 3x3 product
                # U indices: i in [a_i_start, a_i_end), j in [a_j_start, a_j_end)
                # V indices: j in [b_j_start, b_j_end), k in [b_k_start, b_k_end)
                # W indices: k in [b_k_start, b_k_end), i in [a_i_start, a_i_end)
                
                # We need to add to U, V, W at base_idx.
                # U[i, base_idx] += coeff * U_schoolbook[i, j]
                # V[j, base_idx] += coeff * V_schoolbook[j, k]
                # W[k, i, base_idx] += coeff * W_schoolbook[k, i]
                
                # Schoolbook for 3x3:
                # For each i_local, j_local, k_local in 0..2:
                # U[i_local, j_local] = 1
                # V[j_local, k_local] = 1
                # W[k_local, i_local] = 1
                
                for i_loc in range(3):
                    for j_loc in range(3):
                        for k_loc in range(3):
                            i = a_i_start + i_loc
                            j = a_j_start + j_loc
                            k = b_k_start + k_loc
                            
                            # Check j match
                            if j != b_j_start + j_loc:
                                continue # Should not happen if blocks are aligned correctly in Strassen
                
                            U[i * m + j, base_idx] += coeff
                            V[j * p + k, base_idx] += coeff
                            W[k * n + i, base_idx] += coeff
                            
                            # Note: W index is k*n + i.
                            # We need to distribute this to the C blocks.
                            # The C block is determined by i and k.
                            # i is in [0,6), k is in [0,6).
                            # We will handle the C coefficients later?
                            # No, W is part of the decomposition.
                            # The W matrix must reflect the C coefficients.
                            
                            # Actually, the term contributes to C with a specific coefficient.
                            # So W[k, i] should be multiplied by the C coefficient.
                            # But the C coefficient depends on which C block (C00, C01, etc.)
                            
                            # So we should pass C_contribs to this function and use them.
                            
                            # Let's refactor:
                            # The term is M = A_comb * B_comb.
                            # M contributes to C blocks with coefficients c_00, c_01, c_10, c_11.
                            # So W[k, i] should be set to c_block * (schoolbook_W[k_loc, i_loc])
                            # where c_block is the coefficient for the C block containing (k, i).
                            
                            # We need to determine which C block (k, i) falls into.
                            c_block_id = get_c_block_id(i, k)
                            c_coeff = get_c_coeff(c_contribs, c_block_id)
                            
                            # The schoolbook W value is 1.
                            # So W[k * n + i, base_idx] += coeff * c_coeff
                            
                            # Wait, I already added to W above.
                            # I should not add to W inside the i,j,k loop if I'm handling C coeffs.
                            # Let's restart the logic.
                            
                            # The term is a single 3x3 matrix product (conceptually).
                            # It is decomposed into 27 rank-1 terms.
                            # For each rank-1 term (i_loc, j_loc, k_loc):
                            # U[i, j] += coeff_A * coeff_B
                            # V[j, k] += coeff_A * coeff_B
                            # W[k, i] += coeff_A * coeff_B * coeff_C_block(i, k)
                            
                            # So:
                            U_val = coeff
                            V_val = coeff
                            W_val = coeff * c_coeff
                            
                            U[i * m + j, base_idx] += U_val
                            V[j * p + k, base_idx] += V_val
                            W[k * n + i, base_idx] += W_val

    def get_block_indices(block_id, mat):
        # block_id: 0:00, 1:01, 2:10, 3:11
        if mat == 'A':
            if block_id == 0: return (0, 3, 0, 3)
            if block_id == 1: return (0, 3, 3, 6)
            if block_id == 2: return (3, 6, 0, 3)
            if block_id == 3: return (3, 6, 3, 6)
        elif mat == 'B':
            if block_id == 0: return (0, 3, 0, 3)
            if block_id == 1: return (0, 3, 3, 6)
            if block_id == 2: return (3, 6, 0, 3)
            if block_id == 3: return (3, 6, 3, 6)
        return (0, 0, 0, 0)

    def get_c_block_id(i, k):
        # i: row of C, k: col of C
        if i < 3 and k < 3: return 0 # C00
        if i < 3 and k >= 3: return 1 # C01
        if i >= 3 and k < 3: return 2 # C10
        if i >= 3 and k >= 3: return 3 # C11
        return -1

    def get_c_coeff(c_contribs, block_id):
        for b_id, coeff in c_contribs:
            if b_id == block_id:
                return coeff
        return 0

    # Define the 7 Strassen terms
    terms = [
        # (A_blocks, B_blocks, C_contribs)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),          # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),          # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),          # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),          # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),          # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),          # M7
    ]
    
    base_idx = 0
    for a_blocks, b_blocks, c_contribs in terms:
        add_term(a_blocks, b_blocks, c_contribs, base_idx)
        base_idx += 27
        
    # Verify
    def verify(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
