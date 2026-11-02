import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We construct a 4x5x5 decomposition of rank 67, then project to 3x5x5.
    # Rank 67 = 49 (4x4x4 Strassen) + 1 (4x1x4) + 16 (4x4x1) + 1 (4x1x1)
    
    # 1. Generate 2x2x2 Strassen decomposition
    # U, V, W of shape (4, 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1, C01-1, C10-1
    U2[:, 0] = [1, 0, 0, 1]
    V2[:, 0] = [1, 0, 0, 1]
    W2[:, 0] = [1, -1, -1, 1] # C00, C10, C01, C11 (k*2+i: 0,1,2,3)
    
    # M2: (A10+A11)B00 -> C10+1, C11-1
    U2[:, 1] = [0, 0, 1, 1]
    V2[:, 1] = [1, 0, 0, 0]
    W2[:, 1] = [0, 1, 0, -1]
    
    # M3: A00(B01-B11) -> C01+1, C11+1
    U2[:, 2] = [1, 0, 0, 0]
    V2[:, 2] = [0, 1, 0, -1]
    W2[:, 2] = [0, 0, 1, 1]
    
    # M4: A11(B10-B00) -> C00+1, C10+1
    U2[:, 3] = [0, 0, 0, 1]
    V2[:, 3] = [0, 0, 1, -1]
    W2[:, 3] = [1, 1, 0, 0]
    
    # M5: (A00+A01)B11 -> C00-1, C01+1
    U2[:, 4] = [1, 1, 0, 0]
    V2[:, 4] = [0, 0, 0, 1]
    W2[:, 4] = [-1, 0, 1, 0]
    
    # M6: (A10-A00)(B00+B01) -> C11+1
    U2[:, 5] = [-1, 0, 1, 0]
    V2[:, 5] = [1, 1, 0, 0]
    W2[:, 5] = [0, 0, 0, 1]
    
    # M7: (A01-A11)(B10+B11) -> C00+1
    U2[:, 6] = [0, 1, 0, -1]
    V2[:, 6] = [0, 0, 1, 1]
    W2[:, 6] = [1, 0, 0, 0]

    # 2. Construct 4x4x4 Strassen decomposition (Rank 49)
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Define block mappings for 4x4 matrices
    # Block (bi, bj) of size 2x2
    # bi, bj in {0, 1}
    # Indices in 4x4 flattened (i*4+j)
    def get_block_indices(bi, bj):
        # bi: 0 (top), 1 (bottom)
        # bj: 0 (left), 1 (right)
        i_start = bi * 2
        j_start = bj * 2
        indices = []
        for di in range(2):
            for dj in range(2):
                i = i_start + di
                j = j_start + dj
                indices.append(i * 4 + j)
        return indices

    # Mapping of M_k to A-blocks and B-blocks
    # M1: A00, A11 and B00, B11
    # M2: A10, A11 and B00
    # M3: A00 and B01, B11
    # M4: A11 and B10, B00
    # M5: A00, A01 and B11
    # M6: A10, A00 and B00, B01
    # M7: A01, A11 and B10, B11
    
    # We need to know which 2x2 sub-blocks of the 4x4 matrix are used for A and B in each M_k.
    # And for each of the 7 scalar mults in the 2x2 Strassen, we place the U2, V2, W2 patterns.
    
    # Let's define the A-blocks and B-blocks for each M_k
    # Note: The 2x2 Strassen U2, V2, W2 assume the 2x2 blocks are arranged as:
    # A = [A00 A01; A10 A11]
    # So U2 index 0,1 correspond to A00, A01? No.
    # U2 is flattened 2x2.
    # U2[0]=A00, U2[1]=A01, U2[2]=A10, U2[3]=A11.
    
    # For M_k, we need to map the 4 entries of U2 to the 16 entries of U4.
    # This depends on which 2x2 blocks of the 4x4 matrix are used.
    # For example, M1 uses A00 (top-left) and A11 (bottom-right).
    # In the 4x4 matrix, A00 is block (0,0), A11 is block (1,1).
    # So U2[0] (A00 of 2x2) maps to block (0,0) of 4x4.
    # U2[3] (A11 of 2x2) maps to block (1,1) of 4x4.
    # U2[1] (A01) is 0. U2[2] (A10) is 0.
    
    # We need a mapping for each M_k:
    # map_A[k][u2_idx] = (bi, bj) of the 4x4 block, or None if 0.
    # map_B[k][v2_idx] = (bi, bj) of the 4x4 block, or None if 0.
    # map_C[k][w2_idx] = (bi, bj) of the 4x4 block, or None if 0.
    
    # Let's hardcode the block mappings based on Strassen definitions.
    # M1: (A00+A11)(B00+B11)
    # A00 -> block(0,0), A11 -> block(1,1)
    # B00 -> block(0,0), B11 -> block(1,1)
    # C00 -> block(0,0), C11 -> block(1,1), C01 -> block(0,1), C10 -> block(1,0)
    
    # We will iterate k from 0 to 6 (M1 to M7)
    # For each k, we have 7 scalar mults (indices 0 to 6 in U2/V2/W2)
    # Total 49 terms.
    
    for k in range(7):
        # Determine block mappings for this M_k
        # This is complex to derive programmatically without a symbolic engine.
        # We will hardcode the non-zero block indices for U, V, W for each M_k.
        
        # M1 (k=0):
        # U: A00(0,0), A11(1,1)
        # V: B00(0,0), B11(1,1)
        # W: C00(0,0), C10(1,0), C01(0,1), C11(1,1)
        
        # Instead of hardcoding all 49, we can use the property that Strassen is recursive.
        # But for 4x4, it's just one level of Strassen on 2x2 blocks.
        
        # Let's write a helper to place a 2x2 vector into a 4x4 vector at a specific block.
        def place_block(vec2, bi, bj, target):
            # vec2 is length 4: [00, 01, 10, 11]
            # bi, bj is the 2x2 block in 4x4
            idxs = get_block_indices(bi, bj)
            for idx, val in zip(idxs, vec2):
                target[idx] += val

        # We need to know for each M_k, which 2x2 blocks of A, B, C are involved.
        # And for each of the 7 scalar mults (s in 0..6), which of the 4 entries in U2[s] are non-zero?
        # Actually, U2[:, s] is the pattern for the s-th scalar mult of the 2x2 Strassen.
        # We just need to know which 4x4 block each of the 4 entries of U2[:, s] maps to.
        
        # This mapping depends on M_k.
        # For M1:
        # Entry 0 (A00) -> Block(0,0)
        # Entry 1 (A01) -> Block(0,1) ? No, M1 only uses A00 and A11.
        # Wait, U2[:, 0] = [1, 0, 0, 1].
        # So Entry 0 is 1, Entry 3 is 1.
        # Entry 0 corresponds to A00. In M1, A00 is the top-left 2x2 block of the 4x4 matrix?
        # Yes, because M1 = (A00+A11)... where A00 is the top-left 2x2 block of the 4x4 matrix.
        
        # So for M1:
        # U2[0] -> Block(0,0)
        # U2[3] -> Block(1,1)
        # U2[1], U2[2] are 0.
        
        # For M2: (A10+A11)B00
        # A10 -> Block(1,0), A11 -> Block(1,1)
        # B00 -> Block(0,0)
        # U2[:, 1] = [0, 0, 1, 1].
        # Entry 2 (A10) -> Block(1,0)
        # Entry 3 (A11) -> Block(1,1)
        
        # We can define for each k (0..6) the mapping of U2 indices (0..3) to 4x4 blocks.
        
        # Mapping for U (A blocks)
        # k=0 (M1): {0: (0,0), 3: (1,1)}
        # k=1 (M2): {2: (1,0), 3: (1,1)}
        # k=2 (M3): {0: (0,0)}
        # k=3 (M4): {3: (1,1)}
        # k=4 (M5): {0: (0,0), 1: (0,1)}
        # k=5 (M6): {0: (0,0), 2: (1,0)}
        # k=6 (M7): {1: (0,1), 3: (1,1)}
        
        # Mapping for V (B blocks)
        # k=0 (M1): {0: (0,0), 3: (1,1)}
        # k=1 (M2): {0: (0,0)}
        # k=2 (M3): {1: (0,1), 3: (1,1)}
        # k=3 (M4): {0: (0,0), 2: (1,0)}
        # k=4 (M5): {3: (1,1)}
        # k=5 (M6): {0: (0,0), 1: (0,1)}
        # k=6 (M7): {2: (1,0), 3: (1,1)}
        
        # Mapping for W (C blocks)
        # k=0 (M1): {0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)}
        # k=1 (M2): {1: (1,0), 3: (1,1)}
        # k=2 (M3): {2: (0,1), 3: (1,1)}
        # k=3 (M4): {0: (0,0), 1: (1,0)}
        # k=4 (M5): {0: (0,0), 2: (0,1)}
        # k=5 (M6): {3: (1,1)}
        # k=6 (M7): {0: (0,0)}
        
        U_maps = [
            {0: (0,0), 3: (1,1)},
            {2: (1,0), 3: (1,1)},
            {0: (0,0)},
            {3: (1,1)},
            {0: (0,0), 1: (0,1)},
            {0: (0,0), 2: (1,0)},
            {1: (0,1), 3: (1,1)}
        ]
        V_maps = [
            {0: (0,0), 3: (1,1)},
            {0: (0,0)},
            {1: (0,1), 3: (1,1)},
            {0: (0,0), 2: (1,0)},
            {3: (1,1)},
            {0: (0,0), 1: (0,1)},
            {2: (1,0), 3: (1,1)}
        ]
        W_maps = [
            {0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)},
            {1: (1,0), 3: (1,1)},
            {2: (0,1), 3: (1,1)},
            {0: (0,0), 1: (1,0)},
            {0: (0,0), 2: (0,1)},
            {3: (1,1)},
            {0: (0,0)}
        ]
        
        for s in range(7): # 7 scalar mults in 2x2 Strassen
            col = k * 7 + s
            # U
            for u_idx, val in enumerate(U2[:, s]):
                if val != 0:
                    if u_idx in U_maps[k]:
                        bi, bj = U_maps[k][u_idx]
                        place_block([val], bi, bj, U4[:, col]) # Wait, place_block takes vec2
                        # Actually, U2[:, s] is the vec2.
                        # We should just place the whole U2[:, s] if the map exists for all?
                        # No, the map is per entry.
                        
            # Let's do it properly:
            U4_col = np.zeros(16)
            V4_col = np.zeros(16)
            W4_col = np.zeros(16)
            
            for u_idx in range(4):
                val = U2[u_idx, s]
                if val != 0 and u_idx in U_maps[k]:
                    bi, bj = U_maps[k][u_idx]
                    idxs = get_block_indices(bi, bj)
                    # The entry u_idx in U2 corresponds to a specific position in the 2x2 block.
                    # u_idx 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                    # So we add val to the corresponding position in the 4x4 block.
                    di = u_idx // 2
                    dj = u_idx % 2
                    i = bi * 2 + di
                    j = bj * 2 + dj
                    U4_col[i * 4 + j] += val
            
            for v_idx in range(4):
                val = V2[v_idx, s]
                if val != 0 and v_idx in V_maps[k]:
                    bi, bj = V_maps[k][v_idx]
                    di = v_idx // 2
                    dj = v_idx % 2
                    i = bi * 2 + di
                    j = bj * 2 + dj
                    V4_col[i * 4 + j] += val
            
            for w_idx in range(4):
                val = W2[w_idx, s]
                if val != 0 and w_idx in W_maps[k]:
                    bi, bj = W_maps[k][w_idx]
                    di = w_idx // 2
                    dj = w_idx % 2
                    i = bi * 2 + di
                    j = bj * 2 + dj
                    W4_col[i * 4 + j] += val
            
            U4[:, col] = U4_col
            V4[:, col] = V4_col
            W4[:, col] = W4_col

    # Now we have 49 terms for the 4x4x4 part.
    # We need to add the other 18 terms for the 4x5x5 decomposition.
    # Total rank 67.
    
    U_total = np.zeros((20, 67), dtype=int) # 4x5
    V_total = np.zeros((25, 67), dtype=int) # 5x5
    W_total = np.zeros((20, 67), dtype=int) # 5x4
    
    # Copy the 49 terms to the first 49 columns.
    # The 4x4 part corresponds to A[0:4, 0:4], B[0:4, 0:4], C[0:4, 0:4]
    # In U_total (4x5), indices i*5+j. i in 0..3, j in 0..3.
    for i in range(4):
        for j in range(4):
            U_total[i*5+j, 0:49] = U4[i*4+j, 0:49]
            
    # In V_total (5x5), indices j*5+k. j in 0..3, k in 0..3.
    for j in range(4):
        for k in range(4):
            V_total[j*5+k, 0:49] = V4[j*4+k, 0:49]
            
    # In W_total (5x4), indices k*4+i. k in 0..3, i in 0..3.
    for k in range(4):
        for i in range(4):
            W_total[k*4+i, 0:49] = W4[k*4+i, 0:49]

    # Add the remaining 18 terms.
    # 1. A[0:4, 4] * B[4, 0:4] (4x1x4) -> Rank 1
    # Term 49
    # U: A[i, 4] for i in 0..3.
    for i in range(4):
        U_total[i*5+4, 49] = 1
    # V: B[4, k] for k in 0..3.
    for k in range(4):
        V_total[4*5+k, 49] = 1
    # W: C[k, i] for k in 0..3, i in 0..3.
    for k in range(4):
        for i in range(4):
            W_total[k*4+i, 49] = 1

    # 2. A[0:4, 0:4] * B[0:4, 4] (4x4x1) -> Rank 16
    # Terms 50..65
    for j in range(4):
        for i in range(4):
            col = 50 + i*4 + j
            U_total[i*5+j, col] = 1
            V_total[j*5+4, col] = 1
            W_total[4*4+i, col] = 1 # k=4, i in 0..3

    # 3. A[0:4, 4] * B[4, 4] (4x1x1) -> Rank 1
    # Term 66
    for i in range(4):
        U_total[i*5+4, 66] = 1
    V_total[4*5+4, 66] = 1
    for i in range(4):
        W_total[4*4+i, 66] = 1

    # Now project to 3x5x5.
    # Keep first 3 rows of A (i in 0..2).
    U_final = U_total[0:15, :]
    
    # Keep all of B.
    V_final = V_total
    
    # Keep first 3 rows of C (i in 0..2) for each k.
    # W_total indices k*4+i.
    # We need k*3+i for the final W?
    # No, the interface says W shape (p*n, rank) = (5*3, 67) = (15, 67).
    # Indices k*3+i.
    W_final = np.zeros((15, 67), dtype=int)
    for k in range(5):
        for i in range(3):
            W_final[k*3+i, :] = W_total[k*4+i, :]

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

if __name__ == "__main__":
    construct()
