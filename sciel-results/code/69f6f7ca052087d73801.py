import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 multiplication (columns 0-3 of B)
    # Rank 49 using Strassen's algorithm applied recursively.
    # We generate the 49 terms.
    
    # Helper to get Strassen 2x2x2 decomposition
    def get_strassen_2x2():
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: A: 00, 01, 10, 11 -> 0, 1, 2, 3
        #          B: 00, 01, 10, 11 -> 0, 1, 2, 3
        #          C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # Term 1: (A00 + A11)(B00 + B11)
        # U: A00(0), A11(3)
        U[0, 0] = 1
        U[3, 0] = 1
        # V: B00(0), B11(3)
        V[0, 0] = 1
        V[3, 0] = 1
        # W: C00(0), C11(3)
        W[0, 0] = 1
        W[3, 0] = 1
        
        # Term 2: (A10 + A11)B00
        # U: A10(2), A11(3)
        U[2, 1] = 1
        U[3, 1] = 1
        # V: B00(0)
        V[0, 1] = 1
        # W: C10(2), C11(3) with -1
        W[2, 1] = 1
        W[3, 1] = -1
        
        # Term 3: A00(B01 - B11)
        # U: A00(0)
        U[0, 2] = 1
        # V: B01(1), B11(3) with -1
        V[1, 2] = 1
        V[3, 2] = -1
        # W: C01(1), C11(3)
        W[1, 2] = 1
        W[3, 2] = 1
        
        # Term 4: A11(B10 - B00)
        # U: A11(3)
        U[3, 3] = 1
        # V: B10(2), B00(0) with -1
        V[2, 3] = 1
        V[0, 3] = -1
        # W: C00(0), C10(2)
        W[0, 3] = 1
        W[2, 3] = 1
        
        # Term 5: (A00 + A01)B11
        # U: A00(0), A01(1)
        U[0, 4] = 1
        U[1, 4] = 1
        # V: B11(3)
        V[3, 4] = 1
        # W: C00(0) with -1, C01(1)
        W[0, 4] = -1
        W[1, 4] = 1
        
        # Term 6: (A10 - A00)(B00 + B01)
        # U: A10(2), A00(0) with -1
        U[2, 5] = 1
        U[0, 5] = -1
        # V: B00(0), B01(1)
        V[0, 5] = 1
        V[1, 5] = 1
        # W: C11(3)
        W[3, 5] = 1
        
        # Term 7: (A01 - A11)(B10 + B11)
        # U: A01(1), A11(3) with -1
        U[1, 6] = 1
        U[3, 6] = -1
        # V: B10(2), B11(3)
        V[2, 6] = 1
        V[3, 6] = 1
        # W: C00(0), C10(2)
        W[0, 6] = 1
        W[2, 6] = 1
        
        return U, V, W

    U_2x2, V_2x2, W_2x2 = get_strassen_2x2()
    
    # Now we need to expand this to 4x4x4.
    # The 4x4x4 multiplication is a 2x2 block multiplication where blocks are 2x2.
    # We use the 7 terms from the block level.
    # Each block term is a product of two 2x2 matrices.
    # We replace each block multiplication with the 7 scalar multiplications from U_2x2, etc.
    
    # Block indices:
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # Strassen terms for blocks:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We will construct U_444, V_444, W_444 of shape (16, 49).
    # Each of the 7 block terms contributes 7 scalar terms.
    
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int) # Wait, V for 4x4x4 is (4*4, 49) = (16, 49)
    W_444 = np.zeros((16, 49), dtype=int)
    
    # Mapping from block index to sub-matrix indices
    # Block 00: rows 0-1, cols 0-1. Indices in 4x4: 0,1,4,5
    # Block 01: rows 0-1, cols 2-3. Indices in 4x4: 2,3,6,7
    # Block 10: rows 2-3, cols 0-1. Indices in 4x4: 8,9,12,13
    # Block 11: rows 2-3, cols 2-3. Indices in 4x4: 10,11,14,15
    
    def block_indices(block_r, block_c):
        # Returns list of 4 indices in the 4x4 flattened matrix
        # row start = block_r * 2, col start = block_c * 2
        r0 = block_r * 2
        c0 = block_c * 2
        idx = []
        for dr in range(2):
            for dc in range(2):
                idx.append((r0 + dr) * 4 + (c0 + dc))
        return idx

    # We define the 7 block terms.
    # Each term is defined by:
    # U_blocks: list of (block_idx, coeff)
    # V_blocks: list of (block_idx, coeff)
    # W_blocks: list of (block_idx, coeff)
    
    terms = []
    
    # M1
    terms.append({
        'U': [(0, 1), (3, 1)], # A00, A11
        'V': [(0, 1), (3, 1)], # B00, B11
        'W': [(0, 1), (3, 1)]  # C00, C11
    })
    # M2
    terms.append({
        'U': [(2, 1), (3, 1)], # A10, A11
        'V': [(0, 1)],         # B00
        'W': [(2, 1), (3, -1)] # C10, C11
    })
    # M3
    terms.append({
        'U': [(0, 1)],         # A00
        'V': [(1, 1), (3, -1)],# B01, B11
        'W': [(1, 1), (3, 1)]  # C01, C11
    })
    # M4
    terms.append({
        'U': [(3, 1)],         # A11
        'V': [(2, 1), (0, -1)],# B10, B00
        'W': [(0, 1), (2, 1)]  # C00, C10
    })
    # M5
    terms.append({
        'U': [(0, 1), (1, 1)], # A00, A01
        'V': [(3, 1)],         # B11
        'W': [(0, -1), (1, 1)] # C00, C01
    })
    # M6
    terms.append({
        'U': [(2, 1), (0, -1)],# A10, A00
        'V': [(0, 1), (1, 1)], # B00, B01
        'W': [(3, 1)]          # C11
    })
    # M7
    terms.append({
        'U': [(1, 1), (3, -1)],# A01, A11
        'V': [(2, 1), (3, 1)], # B10, B11
        'W': [(0, 1), (2, 1)]  # C00, C10
    })

    col = 0
    for term in terms:
        # For each of the 7 scalar multiplications in the 2x2x2 decomposition
        for s in range(7):
            # Get the U, V, W entries for this scalar term from U_2x2, etc.
            # U_2x2 has shape (4, 7). Column s.
            # Non-zero entries in U_2x2[:, s]
            u_nonzero = np.nonzero(U_2x2[:, s])[0]
            v_nonzero = np.nonzero(V_2x2[:, s])[0]
            w_nonzero = np.nonzero(W_2x2[:, s])[0]
            
            # Combine with block coefficients
            # The scalar term s corresponds to a product of linear combinations of A-block entries
            # and B-block entries.
            
            # We need to sum contributions from all block pairs in term['U'] and term['V']
            # This is effectively a tensor product of the block structure and the internal structure.
            
            # For U:
            # For each (block_idx, coeff) in term['U']:
            #   For each internal_idx in u_nonzero:
            #     U_444[block_indices(block_idx)[internal_idx], col] += coeff * U_2x2[internal_idx, s]
            
            for (ub, uc) in term['U']:
                u_indices = block_indices(ub // 2, ub % 2) # ub is 0..3, representing block row/col
                # Wait, block indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
                # My block_indices function takes (block_r, block_c).
                # ub=0 -> (0,0), ub=1 -> (0,1), ub=2 -> (1,0), ub=3 -> (1,1)
                u_r = ub // 2
                u_c = ub % 2
                u_indices = block_indices(u_r, u_c)
                
                for internal_idx in u_nonzero:
                    val = uc * U_2x2[internal_idx, s]
                    U_444[u_indices[internal_idx], col] += val
            
            for (vb, vc) in term['V']:
                v_r = vb // 2
                v_c = vb % 2
                v_indices = block_indices(v_r, v_c)
                
                for internal_idx in v_nonzero:
                    val = vc * V_2x2[internal_idx, s]
                    V_444[v_indices[internal_idx], col] += val
            
            for (wb, wc) in term['W']:
                w_r = wb // 2
                w_c = wb % 2
                w_indices = block_indices(w_r, w_c)
                
                for internal_idx in w_nonzero:
                    val = wc * W_2x2[internal_idx, s]
                    W_444[w_indices[internal_idx], col] += val
            
            col += 1
    
    # Part 2: 4x4x1 multiplication (column 4 of B)
    # Rank 16.
    U_441 = np.zeros((16, 16), dtype=int)
    V_441 = np.zeros((4, 16), dtype=int) # m*p = 4*1 = 4
    W_441 = np.zeros((4, 16), dtype=int) # p*n = 1*4 = 4
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U_441[i * 4 + j, r] = 1
            V_441[j, r] = 1 # j is row of B, k=0 is col
            W_441[i, r] = 1 # k=0, i is row of C
    
    # Combine
    R = 49 + 16
    U_total = np.zeros((16, R), dtype=int)
    V_total = np.zeros((20, R), dtype=int)
    W_total = np.zeros((20, R), dtype=int)
    
    # Copy Part 1
    U_total[:, :49] = U_444
    # Map V_444 (16, 49) to V_total (20, 49)
    # V_444 index j*4+k. V_total index j*5+k.
    for j in range(4):
        for k in range(4):
            V_total[j * 5 + k, :49] = V_444[j * 4 + k, :49]
    # W_444 (16, 49) to W_total (20, 49)
    # W_444 index k*4+i. W_total index k*4+i.
    # k is 0..3.
    for k in range(4):
        for i in range(4):
            W_total[k * 4 + i, :49] = W_444[k * 4 + i, :49]
    
    # Copy Part 2
    U_total[:, 49:] = U_441
    # V_441 (4, 16) to V_total (20, 16)
    # V_441 index j. V_total index j*5+4.
    for j in range(4):
        V_total[j * 5 + 4, 49:] = V_441[j, :]
    # W_441 (4, 16) to W_total (20, 16)
    # W_441 index i. W_total index 4*4+i = 16+i.
    for i in range(4):
        W_total[16 + i, 49:] = W_441[i, :]
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
