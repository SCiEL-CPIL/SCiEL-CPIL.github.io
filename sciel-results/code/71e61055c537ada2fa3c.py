import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = 5, 5, 5
    
    # Partition 5 = 4 + 1
    block_sizes = [4, 1]
    offsets = [0, 4, 5]
    
    # Strassen factors for 2x2x2
    def get_strassen_2x2():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    strassen_2x2 = get_strassen_2x2()
    
    # Generate Strassen factors for 4x4x4 by recursion
    # A 4x4 matrix is a 2x2 block of 2x2 matrices.
    # Indices for 4x4: 0..15. 
    # Block (i,j) of 2x2 blocks covers indices:
    # (0,0): 0,1,4,5
    # (0,1): 2,3,6,7
    # (1,0): 8,9,12,13
    # (1,1): 10,11,14,15
    # Wait, row-major order for 4x4:
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # Row 3: 12,13,14,15
    #
    # Block A00 (top-left 2x2): rows 0,1 cols 0,1 -> indices 0,1,4,5
    # Block A01 (top-right 2x2): rows 0,1 cols 2,3 -> indices 2,3,6,7
    # Block A10 (bottom-left 2x2): rows 2,3 cols 0,1 -> indices 8,9,12,13
    # Block A11 (bottom-right 2x2): rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    def get_strassen_4x4():
        factors_4x4 = []
        # For each of the 7 top-level Strassen factors
        for (u_top, v_top, w_top) in strassen_2x2:
            # u_top, v_top, w_top are vectors of length 4 (representing 2x2 blocks)
            # Each element is a coefficient for a 2x2 block.
            # We need to expand each "block multiplication" into 7 rank-1 terms.
            
            # The top-level factor computes:
            # M_top = (sum u_top[i] A_block_i) * (sum v_top[j] B_block_j)
            # This is a 2x2 matrix multiplication of the resulting 2x2 blocks.
            # Let X = sum u_top[i] A_block_i (a 2x2 matrix of 2x2 blocks? No, X is a 2x2 matrix)
            # Actually, u_top[i] is a scalar. A_block_i is a 2x2 matrix.
            # So X is a 2x2 matrix. Y is a 2x2 matrix.
            # M_top = X * Y.
            # We decompose X*Y using the 2x2 Strassen factors.
            
            # X is formed by linear combination of A_blocks.
            # Y is formed by linear combination of B_blocks.
            # W is formed by linear combination of C_blocks.
            
            # For each of the 7 bottom-level Strassen factors (u_bot, v_bot, w_bot):
            for (u_bot, v_bot, w_bot) in strassen_2x2:
                # u_bot, v_bot, w_bot are length 4 vectors for 2x2 matrices.
                # Indices 0,1,2,3 correspond to 2x2 matrix entries [0,0], [0,1], [1,0], [1,1].
                
                U_vec = [0.0] * 16
                V_vec = [0.0] * 16
                W_vec = [0.0] * 16
                
                # Map 2x2 block indices to 4x4 global indices
                # Block 0 (0,0): 0,1,4,5
                # Block 1 (0,1): 2,3,6,7
                # Block 2 (1,0): 8,9,12,13
                # Block 3 (1,1): 10,11,14,15
                
                # For U:
                # u_bot[k] is coefficient for X_block_k.
                # X_block_k = sum_i u_top[i] A_block_i.
                # So U_vec gets contributions from all A_blocks.
                # Specifically, for each k in 0..3 (index in 2x2 X):
                #   coeff_X_k = u_bot[k]
                #   X_k = sum_i u_top[i] A_i
                #   So term is coeff_X_k * u_top[i] * A_i
                #   We add this to U_vec at positions corresponding to A_i.
                
                # We need to map the 2x2 block indices (0,1,2,3) to the 4 indices of the 2x2 blocks in the 4x4 matrix.
                # Let block_idx_map = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]
                # This map tells us the 4 global indices for each 2x2 block.
                
                block_indices = [
                    [0, 1, 4, 5],    # Block 0
                    [2, 3, 6, 7],    # Block 1
                    [8, 9, 12, 13],  # Block 2
                    [10, 11, 14, 15] # Block 3
                ]
                
                for k in range(4): # k is index in 2x2 matrix X
                    coeff_X = u_bot[k]
                    if coeff_X == 0: continue
                    for i in range(4): # i is index of A_block
                        coeff_A = u_top[i]
                        if coeff_A == 0: continue
                        val = coeff_X * coeff_A
                        # Add val to U_vec at indices of A_block i
                        for idx in block_indices[i]:
                            U_vec[idx] += val
                            
                for k in range(4): # k is index in 2x2 matrix Y
                    coeff_Y = v_bot[k]
                    if coeff_Y == 0: continue
                    for j in range(4): # j is index of B_block
                        coeff_B = v_top[j]
                        if coeff_B == 0: continue
                        val = coeff_Y * coeff_B
                        for idx in block_indices[j]:
                            V_vec[idx] += val
                            
                for k in range(4): # k is index in 2x2 matrix Z (result)
                    coeff_Z = w_bot[k]
                    if coeff_Z == 0: continue
                    for l in range(4): # l is index of C_block
                        coeff_C = w_top[l]
                        if coeff_C == 0: continue
                        val = coeff_Z * coeff_C
                        for idx in block_indices[l]:
                            W_vec[idx] += val
                            
                factors_4x4.append((U_vec, V_vec, W_vec))
                
        return factors_4x4

    strassen_4x4 = get_strassen_4x4()
    
    # Total rank calculation
    # Block (0,0,0): 4x4x4 -> 49
    # Block (0,0,1): 4x4x1 -> 16
    # Block (0,1,0): 4x1x4 -> 16
    # Block (0,1,1): 4x1x1 -> 4
    # Block (1,0,0): 1x4x4 -> 16
    # Block (1,0,1): 1x4x1 -> 4
    # Block (1,1,0): 1x1x4 -> 4
    # Block (1,1,1): 1x1x1 -> 1
    total_rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1
    total_rank = 110
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook factors
    def add_schoolbook(ni, mj, pk, off_i, off_j, off_k):
        nonlocal col
        for li in range(ni):
            for lj in range(mj):
                for lk in range(pk):
                    U_idx = (off_i + li) * m + (off_j + lj)
                    V_idx = (off_j + lj) * p + (off_k + lk)
                    W_idx = (off_k + lk) * n + (off_i + li)
                    U[U_idx][col] = 1.0
                    V[V_idx][col] = 1.0
                    W[W_idx][col] = 1.0
                    col += 1
    
    # Helper to add Strassen 4x4 factors
    def add_strassen_4x4(off_i, off_j, off_k):
        nonlocal col
        # off_i, off_j, off_k are the starting row/col indices for the 4x4 blocks
        # Since block size is 4, the indices are off_i..off_i+3 etc.
        # The global indices for U (n*m) are (row, col) where row in 0..4, col in 0..4.
        # For the 4x4 block at (0,0), rows 0..3, cols 0..3.
        # U indices: (r, c) -> r*m + c.
        # Here r is in 0..3, c is in 0..3.
        # But wait, the block is at offset (off_i, off_j).
        # So global row = off_i + r, global col = off_j + c.
        # U global index = (off_i + r) * m + (off_j + c).
        
        for (u_vec, v_vec, w_vec) in strassen_4x4:
            # u_vec, v_vec, w_vec are length 16.
            # Index in vec corresponds to local (r, c) in 4x4.
            # local index = r*4 + c.
            for idx in range(16):
                r = idx // 4
                c = idx % 4
                
                if u_vec[idx] != 0:
                    U_global = (off_i + r) * m + (off_j + c)
                    U[U_global][col] += u_vec[idx]
                
                if v_vec[idx] != 0:
                    V_global = (off_j + r) * p + (off_k + c)
                    V[V_global][col] += v_vec[idx]
                
                if w_vec[idx] != 0:
                    W_global = (off_k + r) * n + (off_i + c)
                    W[W_global][col] += w_vec[idx]
            
            col += 1
    
    # Block (0,0,0): 4x4x4
    add_strassen_4x4(0, 0, 0)
    
    # Block (0,0,1): 4x4x1
    add_schoolbook(4, 4, 1, 0, 0, 4)
    
    # Block (0,1,0): 4x1x4
    add_schoolbook(4, 1, 4, 0, 4, 0)
    
    # Block (0,1,1): 4x1x1
    add_schoolbook(4, 1, 1, 0, 4, 4)
    
    # Block (1,0,0): 1x4x4
    add_schoolbook(1, 4, 4, 4, 0, 0)
    
    # Block (1,0,1): 1x4x1
    add_schoolbook(1, 4, 1, 4, 0, 4)
    
    # Block (1,1,0): 1x1x4
    add_schoolbook(1, 1, 4, 4, 4, 0)
    
    # Block (1,1,1): 1x1x1
    add_schoolbook(1, 1, 1, 4, 4, 4)
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U, "V": V, "W": W}
