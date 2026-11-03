import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We decompose p=9 into 4+4+1
    # Rank 4x4x4 is 49 (Strassen)
    # Rank 4x4x1 is 16
    # Total rank 49 + 49 + 16 = 114
    
    rank_444 = 49
    rank_441 = 16
    total_rank = 2 * rank_444 + rank_441
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Helper to generate Strassen decomposition for n x n x n
    def get_strassen_tensors(n):
        if n == 2:
            # Base case: 7 tensors
            # u, v, w are lists of (row, col) or just values in flattened vectors
            # We return lists of (u_vec, v_vec, w_vec) as numpy arrays of length n*n
            tensors = []
            # M1
            u1 = np.zeros(4); u1[0] = 1; u1[3] = 1
            v1 = np.zeros(4); v1[0] = 1; v1[3] = 1
            w1 = np.zeros(4); w1[0] = 1; w1[3] = 1 # C00, C11
            tensors.append((u1, v1, w1))
            
            # M2
            u2 = np.zeros(4); u2[2] = 1; u2[3] = 1
            v2 = np.zeros(4); v2[0] = 1
            w2 = np.zeros(4); w2[2] = 1; w2[3] = -1 # C10, C11
            tensors.append((u2, v2, w2))
            
            # M3
            u3 = np.zeros(4); u3[0] = 1
            v3 = np.zeros(4); v3[1] = 1; v3[3] = -1
            w3 = np.zeros(4); w3[1] = 1; w3[3] = 1 # C01, C11
            tensors.append((u3, v3, w3))
            
            # M4
            u4 = np.zeros(4); u4[3] = 1
            v4 = np.zeros(4); v4[2] = 1; v4[0] = -1
            w4 = np.zeros(4); w4[0] = 1; w4[2] = 1 # C00, C10
            tensors.append((u4, v4, w4))
            
            # M5
            u5 = np.zeros(4); u5[0] = 1; u5[1] = 1
            v5 = np.zeros(4); v5[3] = 1
            w5 = np.zeros(4); w5[0] = -1; w5[1] = 1 # C00, C01
            tensors.append((u5, v5, w5))
            
            # M6
            u6 = np.zeros(4); u6[2] = 1; u6[0] = -1
            v6 = np.zeros(4); v6[0] = 1; v6[1] = 1
            w6 = np.zeros(4); w6[3] = 1 # C11
            tensors.append((u6, v6, w6))
            
            # M7
            u7 = np.zeros(4); u7[1] = 1; u7[3] = -1
            v7 = np.zeros(4); v7[2] = 1; v7[3] = 1
            w7 = np.zeros(4); w7[0] = 1 # C00
            tensors.append((u7, v7, w7))
            
            return tensors
        else:
            # Recursive case
            half = n // 2
            sub_tensors = get_strassen_tensors(half)
            
            # We need to combine 7 sub-products
            # Each sub-product is a half x half matrix multiplication
            # The u, v vectors are in the space of half x half matrices
            # We need to map them to the full n x n space
            
            full_tensors = []
            
            # Define the 7 combinations for u and v
            # A blocks: A00, A01, A10, A11
            # B blocks: B00, B01, B10, B11
            
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # For each of the 7 M's, we have a linear combination of A blocks and B blocks.
            # We iterate through the 7 sub-tensors of the base case?
            # No, the sub-tensors are for the multiplication of the blocks.
            # So for M1, we multiply (A00+A11) by (B00+B11).
            # This is a matrix multiplication of size half.
            # So we use the sub_tensors (which are rank-1 decompositions of the half-sized MM).
            
            # For each sub-tensor (u_sub, v_sub, w_sub) in sub_tensors:
            # This represents a rank-1 term in the multiplication of two half-sized matrices.
            # We need to apply this to the specific linear combinations for M1..M7.
            
            # Actually, it's easier to think:
            # The decomposition of M_n is the sum of 7 decompositions of M_{n/2}.
            # Each M_{n/2} decomposition has rank R_{n/2}.
            # So total rank is 7 * R_{n/2}.
            
            # For each of the 7 Strassen terms (M1..M7):
            #   For each rank-1 tensor (u_sub, v_sub, w_sub) in the decomposition of M_{n/2}:
            #       Construct u_full, v_full, w_full.
            
            for m_idx in range(7):
                # Define u_comb and v_comb as linear combinations of block identities
                # u_comb is a 4x4 matrix (conceptually) or flattened
                # We need to map the sub-tensor's u_sub (length half*half) to the full u (length n*n)
                
                # Let's define the block offsets
                # A00: 0..half-1, 0..half-1
                # A01: 0..half-1, half..n-1
                # A10: half..n-1, 0..half-1
                # A11: half..n-1, half..n-1
                
                # For M1: u is A00 + A11.
                # So u_full has u_sub at A00 block and u_sub at A11 block.
                
                u_combinations = [
                    [(0,0), (1,1)], # M1: A00, A11
                    [(1,0), (1,1)], # M2: A10, A11
                    [(0,0)],        # M3: A00
                    [(1,1)],        # M4: A11
                    [(0,0), (0,1)], # M5: A00, A01
                    [(1,0), (0,0)], # M6: A10, -A00 (coeff -1)
                    [(0,1), (1,1)]  # M7: A01, -A11 (coeff -1)
                ]
                
                # Note: M6 and M7 have negative coefficients for one block.
                # We need to handle coefficients.
                
                # Let's re-define with coefficients
                # M1: A00(1) + A11(1)
                # M2: A10(1) + A11(1)
                # M3: A00(1)
                # M4: A11(1)
                # M5: A00(1) + A01(1)
                # M6: A10(1) + A00(-1)
                # M7: A01(1) + A11(-1)
                
                u_defs = [
                    { (0,0): 1, (1,1): 1 },
                    { (1,0): 1, (1,1): 1 },
                    { (0,0): 1 },
                    { (1,1): 1 },
                    { (0,0): 1, (0,1): 1 },
                    { (1,0): 1, (0,0): -1 },
                    { (0,1): 1, (1,1): -1 }
                ]
                
                v_defs = [
                    { (0,0): 1, (1,1): 1 }, # M1: B00+B11
                    { (0,0): 1 },            # M2: B00
                    { (0,1): 1, (1,1): -1 }, # M3: B01-B11
                    { (1,0): 1, (0,0): -1 }, # M4: B10-B00
                    { (1,1): 1 },            # M5: B11
                    { (0,0): 1, (0,1): 1 },  # M6: B00+B01
                    { (1,0): 1, (1,1): 1 }   # M7: B10+B11
                ]
                
                w_defs = [
                    { (0,0): 1, (1,1): 1 }, # M1: C00+C11
                    { (1,0): 1, (1,1): -1 }, # M2: C10-C11
                    { (0,1): 1, (1,1): 1 },  # M3: C01+C11
                    { (0,0): 1, (1,0): 1 },  # M4: C00+C10
                    { (0,0): -1, (0,1): 1 }, # M5: -C00+C01
                    { (1,1): 1 },            # M6: C11
                    { (0,0): 1 }             # M7: C00
                ]
                
                for u_sub, v_sub, w_sub in sub_tensors:
                    u_full = np.zeros(n*n)
                    v_full = np.zeros(n*n)
                    w_full = np.zeros(n*n)
                    
                    # Fill u_full
                    for (r, c), coeff in u_defs[m_idx].items():
                        # Block (r,c) is at row r*half, col c*half
                        # Indices in flattened n*n: (r*half + i) * n + (c*half + j)
                        # u_sub is flattened half*half: i*half + j
                        for i in range(half):
                            for j in range(half):
                                idx_sub = i * half + j
                                idx_full = (r * half + i) * n + (c * half + j)
                                u_full[idx_full] += coeff * u_sub[idx_sub]
                                
                    # Fill v_full
                    for (r, c), coeff in v_defs[m_idx].items():
                        for i in range(half):
                            for j in range(half):
                                idx_sub = i * half + j
                                idx_full = (r * half + i) * n + (c * half + j)
                                v_full[idx_full] += coeff * v_sub[idx_sub]
                                
                    # Fill w_full
                    for (r, c), coeff in w_defs[m_idx].items():
                        for i in range(half):
                            for j in range(half):
                                idx_sub = i * half + j
                                idx_full = (r * half + i) * n + (c * half + j)
                                w_full[idx_full] += coeff * w_sub[idx_sub]
                                
                    full_tensors.append((u_full, v_full, w_full))
                    
            return full_tensors

    # Get tensors for 4x4x4
    tensors_444 = get_strassen_tensors(4)
    
    # Add first 4x4x4 block (cols 0-3)
    for u, v, w in tensors_444:
        # U: 16 x 1
        U[:, col_offset] = u
        # V: 36 x 1. Map k (0-3) to 0-3
        for j in range(4):
            for k in range(4):
                V[j * 9 + k, col_offset] = v[j * 4 + k]
        # W: 36 x 1. Map k (0-3) to 0-3
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, col_offset] = w[k * 4 + i]
        col_offset += 1
        
    # Add second 4x4x4 block (cols 4-7)
    for u, v, w in tensors_444:
        U[:, col_offset] = u
        for j in range(4):
            for k in range(4):
                V[j * 9 + (4 + k), col_offset] = v[j * 4 + k]
        for k in range(4):
            for i in range(4):
                W[(4 + k) * 4 + i, col_offset] = w[k * 4 + i]
        col_offset += 1
        
    # Add 4x4x1 block (col 8)
    # Rank 16
    for i in range(4):
        for j in range(4):
            u = np.zeros(16)
            u[i * 4 + j] = 1
            V[j * 9 + 8, col_offset] = 1
            W[8 * 4 + i, col_offset] = 1
            U[:, col_offset] = u
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
