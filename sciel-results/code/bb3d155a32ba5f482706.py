import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Generate Strassen decomposition for 4x4 matrix multiplication
    # This will produce rank 49 factors for 4x4 * 4x4 -> 4x4
    def get_strassen_factors(size):
        if size == 1:
            # U: (1, 1), V: (1, 1), W: (1, 1)
            # Flat vectors of length 1
            return [[1]], [[1]], [[1]]
        
        half = size // 2
        # Get factors for half size
        U_half, V_half, W_half = get_strassen_factors(half)
        
        U_full = []
        V_full = []
        W_full = []
        
        # Strassen combinations for 2x2 blocks
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for A blocks (u_coeffs) and B blocks (v_coeffs)
        # Block indices: 00, 01, 10, 11
        # A00, A01, A10, A11
        # B00, B01, B10, B11
        
        # For each term, we have linear combinations of blocks
        # We need to map block indices to flat indices of the larger matrix
        
        def block_to_flat(block_idx, size, half):
            # block_idx is (r, c) where r,c in {0,1}
            # Returns list of flat indices for the submatrix
            # Submatrix is half x half
            # Flat index in size x size matrix: i * size + j
            # i = r * half + i_sub
            # j = c * half + j_sub
            indices = []
            for i_sub in range(half):
                for j_sub in range(half):
                    i = block_idx[0] * half + i_sub
                    j = block_idx[1] * half + j_sub
                    indices.append(i * size + j)
            return indices

        # Define the 7 terms
        # Each term is a list of (block_a, coeff_a, block_b, coeff_b)
        # But actually, it's (A_combo, B_combo)
        # We will iterate through the 7 Strassen terms
        
        terms = [
            # (A_blocks, B_blocks)
            # A_blocks: list of (block_idx, coeff)
            # B_blocks: list of (block_idx, coeff)
            [
                ([( (0,0), 1), ((1,1), 1)], [( (0,0), 1), ((1,1), 1)]),
                ([( (1,0), 1), ((1,1), 1)], [( (0,0), 1)]),
                ([( (0,0), 1)], [( (0,1), 1), ((1,1), -1)]),
                ([( (1,1), 1)], [( (1,0), 1), ((0,0), -1)]),
                ([( (0,0), 1), ((0,1), 1)], [( (1,1), 1)]),
                ([( (1,0), 1), ((0,0), -1)], [( (0,0), 1), ((0,1), 1)]),
                ([( (0,1), 1), ((1,1), -1)], [( (1,0), 1), ((1,1), 1)])
            ]
        ]
        
        # This structure is getting complex. Let's simplify.
        # We have 7 block multiplications.
        # For each block multiplication, we use the half-size factors.
        # The A block is a linear combination of half-size matrices.
        # The B block is a linear combination of half-size matrices.
        
        # Let's define the 7 combinations explicitly as coefficients for the 4 blocks.
        # A_combos: list of 7 lists of 4 coeffs (for A00, A01, A10, A11)
        # B_combos: list of 7 lists of 4 coeffs (for B00, B01, B10, B11)
        
        A_combos = [
            [1, 0, 0, 1], # M1: A00 + A11
            [0, 0, 1, 1], # M2: A10 + A11
            [1, 0, 0, 0], # M3: A00
            [0, 0, 0, 1], # M4: A11
            [1, 1, 0, 0], # M5: A00 + A01
            [-1, 0, 1, 0],# M6: A10 - A00
            [0, 1, -1, 0] # M7: A01 - A11
        ]
        
        B_combos = [
            [1, 0, 0, 1], # M1: B00 + B11
            [1, 0, 0, 0], # M2: B00
            [0, 1, 0, -1],# M3: B01 - B11
            [0, 0, 1, -1],# M4: B10 - B00
            [0, 0, 0, 1], # M5: B11
            [1, 1, 0, 0], # M6: B00 + B01
            [0, 0, 1, 1]  # M7: B10 + B11
        ]
        
        # C_combos: How each M_k contributes to C00, C01, C10, C11
        C_combos = [
            [1, 0, 0, 1], # M1: C00 +=, C11 +=
            [0, 0, 1, -1],# M2: C10 +=, C11 -=
            [0, 1, 0, 1], # M3: C01 +=, C11 +=
            [1, 0, 1, 0], # M4: C00 +=, C10 +=
            [-1, 1, 0, 0],# M5: C00 -=, C01 +=
            [0, 0, 0, 1], # M6: C11 +=
            [1, 0, 0, 0]  # M7: C00 +=
        ]
        
        # For each of the 7 terms:
        for t in range(7):
            # Get half-size factors
            U_h, V_h, W_h = get_strassen_factors(half)
            
            # For each half-size factor r:
            for r in range(len(U_h)):
                u_flat = [0] * (size * size)
                v_flat = [0] * (size * size)
                w_flat = [0] * (size * size)
                
                # Fill u_flat based on A_combos[t]
                for b_idx in range(4):
                    coeff = A_combos[t][b_idx]
                    if coeff != 0:
                        # Get flat indices for this block
                        block_r = b_idx // 2
                        block_c = b_idx % 2
                        indices = block_to_flat((block_r, block_c), size, half)
                        for idx in indices:
                            # u_h[r] is the coefficient for this position in the half matrix
                            # The position in the half matrix corresponds to idx in the full matrix
                            # We need to map the half matrix index to the full matrix index
                            # Actually, block_to_flat returns the full matrix indices.
                            # u_h[r] is a vector of length half*half.
                            # We need to distribute u_h[r] into u_flat at the positions of the block.
                            # But u_h[r] corresponds to the specific submatrix.
                            # The indices returned are in the order i_sub, j_sub.
                            # u_h[r] is also in that order (flat).
                            # So we can just add coeff * u_h[r][k] to u_flat[indices[k]]
                            pass # Need to fix this logic
                
                # Correct logic:
                u_flat = [0] * (size * size)
                v_flat = [0] * (size * size)
                w_flat = [0] * (size * size)
                
                for b_idx in range(4):
                    coeff_a = A_combos[t][b_idx]
                    coeff_b = B_combos[t][b_idx]
                    
                    if coeff_a != 0:
                        block_r = b_idx // 2
                        block_c = b_idx % 2
                        indices = block_to_flat((block_r, block_c), size, half)
                        for k, idx in enumerate(indices):
                            u_flat[idx] += coeff_a * U_h[r][k]
                            
                    if coeff_b != 0:
                        block_r = b_idx // 2
                        block_c = b_idx % 2
                        indices = block_to_flat((block_r, block_c), size, half)
                        for k, idx in enumerate(indices):
                            v_flat[idx] += coeff_b * V_h[r][k]
                
                # Fill w_flat based on C_combos[t]
                # C_combos[t] gives coefficients for C00, C01, C10, C11
                # W_h[r] is the factor for the half-size C matrix.
                # We need to distribute W_h[r] into the C blocks.
                # Note: W in the tensor is indexed by (k, i).
                # In the half-size problem, W_h is for C_half.
                # The C blocks are C00, C01, C10, C11.
                # Each is half x half.
                # The indices in W_flat are (k, i) in the full size x size matrix.
                # k is row, i is col.
                # Block C00: k in 0..half-1, i in 0..half-1.
                # Block C01: k in 0..half-1, i in half..2*half-1.
                # Block C10: k in half..2*half-1, i in 0..half-1.
                # Block C11: k in half..2*half-1, i in half..2*half-1.
                
                for c_idx in range(4):
                    coeff_c = C_combos[t][c_idx]
                    if coeff_c != 0:
                        block_r = c_idx // 2 # k block
                        block_c = c_idx % 2 # i block
                        indices = block_to_flat((block_r, block_c), size, half)
                        for k, idx in enumerate(indices):
                            w_flat[idx] += coeff_c * W_h[r][k]
                            
                U_full.append(u_flat)
                V_full.append(v_flat)
                W_full.append(w_flat)
                
        return U_full, V_full, W_full

    U_4, V_4, W_4 = get_strassen_factors(4)
    rank_4 = len(U_4)
    
    # Construct 4x4x8 decomposition
    # Duplicate the 4x4 factors
    # U: 16 rows, 98 cols
    # V: 32 rows, 98 cols
    # W: 32 rows, 98 cols
    
    rank = rank_4 * 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First half: columns 0..3 of B, rows 0..3 of C
    # Second half: columns 4..7 of B, rows 4..7 of C
    
    for r in range(rank_4):
        # First copy
        for i in range(n * m):
            U[i][r] = U_4[r][i]
        for j in range(m):
            for k in range(4):
                V[j * p + k][r] = V_4[r][j * 4 + k]
        for k in range(4):
            for i in range(n):
                W[k * n + i][r] = W_4[r][k * n + i]
                
        # Second copy
        r2 = r + rank_4
        for i in range(n * m):
            U[i][r2] = U_4[r][i]
        for j in range(m):
            for k in range(4, 8):
                V[j * p + k][r2] = V_4[r][j * 4 + (k - 4)]
        for k in range(4, 8):
            for i in range(n):
                W[k * n + i][r2] = W_4[r][(k - 4) * n + i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
