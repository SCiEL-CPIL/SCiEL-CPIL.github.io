"""Rewritten program for matrix multiplication 4,5,5 using Strassen-Winograd decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = 49 (for 4x4x4 part) + 20 (for j=4 part) = 69
    rank = 69
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Winograd 2x2 decomposition
    # u, v, w are 4-vectors corresponding to 2x2 matrices (row-major)
    winograd_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 1, 0, 0],
        [0, 0, 1, -1]
    ]
    winograd_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    winograd_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # 2. Strassen 4x4 definitions
    # Each term k defines X_k and Y_k as sums of 2x2 blocks of A and B.
    # A blocks: 00, 01, 10, 11. Indices in 4x4 A:
    # 00: [0,1,4,5], 01: [2,3,6,7], 10: [8,9,12,13], 11: [10,11,14,15]
    # We represent block coefficients as dict {block_name: coeff}
    
    strassen_terms = [
        # (X_blocks, Y_blocks)
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}), # M1
        ({'10': 1, '11': 1}, {'00': 1}),           # M2
        ({'00': 1}, {'10': 1, '11': -1}),          # M3
        ({'11': 1}, {'01': 1, '00': -1}),          # M4
        ({'00': 1, '01': 1}, {'11': 1}),           # M5
        ({'01': 1, '00': -1}, {'00': 1, '10': 1}), # M6
        ({'10': 1, '11': -1}, {'11': 1, '01': 1})  # M7
    ]
    
    # Mapping from block name to indices in 4x4 matrix (row-major)
    block_indices = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    
    # C block mapping for Strassen
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    c_block_coeffs = {
        '00': {0: 1, 3: 1, 4: -1, 6: 1},
        '01': {2: 1, 4: 1},
        '10': {1: 1, 3: 1},
        '11': {0: 1, 1: -1, 2: 1, 5: 1}
    }
    
    col = 0
    for k in range(7):
        x_blocks = strassen_terms[k][0]
        y_blocks = strassen_terms[k][1]
        
        # Construct 4 vectors for X_k (each 16-dim)
        # X_k is a 2x2 matrix. Entries X_00, X_01, X_10, X_11.
        # Each entry is a linear combo of A entries.
        x_entries = []
        for pos in range(4): # 0:00, 1:01, 2:10, 3:11
            vec = [0.0] * 16
            for block_name, coeff in x_blocks.items():
                idxs = block_indices[block_name]
                # Map block position (2x2) to local 2x2 position
                # pos 0 -> 00, pos 1 -> 01, pos 2 -> 10, pos 3 -> 11
                # The block_name refers to the 4x4 block.
                # The entry pos refers to the 2x2 position within the block sum.
                # Wait, X_k is a sum of 4x4 blocks? No, X_k is a 2x2 matrix of 2x2 blocks.
                # So X_k[0,0] is a 2x2 block.
                # The block_indices give the 4 indices in the 16-dim vector for a 2x2 block.
                # We need to add coeff * vec for each index.
                for i in idxs:
                    vec[i] += coeff
            x_entries.append(vec)
            
        y_entries = []
        for pos in range(4):
            vec = [0.0] * 16
            for block_name, coeff in y_blocks.items():
                idxs = block_indices[block_name]
                for i in idxs:
                    vec[i] += coeff
            y_entries.append(vec)
            
        # Apply Winograd decomposition
        for t in range(7):
            u_t = winograd_u[t]
            v_t = winograd_v[t]
            w_t = winograd_w[t]
            
            # Compute A vector (16-dim)
            a_vec = [0.0] * 16
            for pos in range(4):
                coeff = u_t[pos]
                if coeff != 0:
                    for i in range(16):
                        a_vec[i] += coeff * x_entries[pos][i]
                        
            # Compute B vector (16-dim)
            b_vec = [0.0] * 16
            for pos in range(4):
                coeff = v_t[pos]
                if coeff != 0:
                    for i in range(16):
                        b_vec[i] += coeff * y_entries[pos][i]
                        
            # Compute C vector (16-dim)
            c_vec = [0.0] * 16
            # w_t is 4-dim for 2x2 C block.
            # We need to map this to the 4x4 C matrix.
            # The C block corresponding to M_k is determined by c_block_coeffs.
            # Actually, M_k contributes to all 4 C blocks with specific coefficients.
            # The Winograd term t produces a 2x2 matrix W_t.
            # This W_t is added to the C blocks with the coefficient from c_block_coeffs.
            # Wait, the Winograd decomposition is for the product of two 2x2 matrices.
            # The result is a 2x2 matrix.
            # In Strassen, M_k is a 2x2 matrix of bilinear forms.
            # The Winograd term t gives a rank-1 contribution to M_k.
            # So we need to add w_t to the C blocks where M_k appears.
            
            for c_block_name, coeffs in c_block_coeffs.items():
                if k in coeffs:
                    scale = coeffs[k]
                    if scale == 0:
                        continue
                    idxs = block_indices[c_block_name]
                    for pos in range(4):
                        w_coeff = w_t[pos]
                        if w_coeff != 0:
                            c_vec[idxs[pos]] += scale * w_coeff

            # Map local indices to global indices
            # Local A (4x4) to Global A (4x5)
            # Local (i,j) -> Global (i, j) since j < 4
            # Global A index: i*5 + j
            # Local B (4x4) to Global B (5x5)
            # Local (j,k) -> Global (j, k) since j,k < 4
            # Global B index: j*5 + k
            # Local C (4x4) to Global C (4x5)
            # Local (i,k) -> Global (i, k)
            # Global C index: k*4 + i
            
            for i in range(4):
                for j in range(4):
                    val = a_vec[i * 4 + j]
                    if val != 0:
                        global_a = i * 5 + j
                        U[global_a][col] = val
                        
            for j in range(4):
                for k in range(4):
                    val = b_vec[j * 4 + k]
                    if val != 0:
                        global_b = j * 5 + k
                        V[global_b][col] = val
                        
            for i in range(4):
                for k in range(4):
                    val = c_vec[i * 4 + k]
                    if val != 0:
                        global_c = k * 4 + i
                        W[global_c][col] = val
                        
            col += 1
            
    # 3. Outer product terms for j=4
    # A_{i,4} B_{4,k} C_{i,k}
    for i in range(4):
        for k in range(5):
            global_a = i * 5 + 4
            global_b = 4 * 5 + k
            global_c = k * 4 + i
            U[global_a][col] = 1
            V[global_b][col] = 1
            W[global_c][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
