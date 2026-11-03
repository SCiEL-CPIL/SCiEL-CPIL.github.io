import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base case: 2x2x2 Strassen decomposition
    # Returns U (4x7), V (4x7), W (4x7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    def base_strassen():
        # Using the standard Strassen 7 multiplications
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # We need to map these to U, V, W factors.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U rows correspond to A elements: (0,0), (0,1), (1,0), (1,1)
        # V rows correspond to B elements: (0,0), (0,1), (1,0), (1,1)
        # W rows correspond to C elements: (0,0), (0,1), (1,0), (1,1)
        
        # Let's construct U, V, W as 4x7 matrices
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A00 + A11)(B00 + B11)
        # A00 is idx 0, A11 is idx 3
        U[0, 0] += 1
        U[3, 0] += 1
        # B00 is idx 0, B11 is idx 3
        V[0, 0] += 1
        V[3, 0] += 1
        # Contributes to C00, C11
        # C00 = k=0, i=0 -> idx 0
        # C11 = k=1, i=1 -> idx 3
        W[0, 0] += 1
        W[3, 0] += 1
        
        # M2: (A10 + A11)B00
        # A10 is idx 2, A11 is idx 3
        U[2, 1] += 1
        U[3, 1] += 1
        # B00 is idx 0
        V[0, 1] += 1
        # Contributes to C10, C11
        # C10 = k=0, i=1 -> idx 1
        # C11 = k=1, i=1 -> idx 3
        W[1, 1] += 1
        W[3, 1] -= 1
        
        # M3: A00(B01 - B11)
        # A00 is idx 0
        U[0, 2] += 1
        # B01 is idx 1, B11 is idx 3
        V[1, 2] += 1
        V[3, 2] -= 1
        # Contributes to C01, C11
        # C01 = k=1, i=0 -> idx 2
        # C11 = k=1, i=1 -> idx 3
        W[2, 2] += 1
        W[3, 2] += 1
        
        # M4: A11(B10 - B00)
        # A11 is idx 3
        U[3, 3] += 1
        # B10 is idx 2, B00 is idx 0
        V[2, 3] += 1
        V[0, 3] -= 1
        # Contributes to C00, C10
        # C00 = k=0, i=0 -> idx 0
        # C10 = k=0, i=1 -> idx 1
        W[0, 3] += 1
        W[1, 3] += 1
        
        # M5: (A00 + A01)B11
        # A00 is idx 0, A01 is idx 1
        U[0, 4] += 1
        U[1, 4] += 1
        # B11 is idx 3
        V[3, 4] += 1
        # Contributes to C01, C00
        # C01 = k=1, i=0 -> idx 2
        # C00 = k=0, i=0 -> idx 0
        W[2, 4] += 1
        W[0, 4] -= 1
        
        # M6: (A10 - A00)(B00 + B01)
        # A10 is idx 2, A00 is idx 0
        U[2, 5] += 1
        U[0, 5] -= 1
        # B00 is idx 0, B01 is idx 1
        V[0, 5] += 1
        V[1, 5] += 1
        # Contributes to C11
        # C11 = k=1, i=1 -> idx 3
        W[3, 5] += 1
        
        # M7: (A01 - A11)(B10 + B11)
        # A01 is idx 1, A11 is idx 3
        U[1, 6] += 1
        U[3, 6] -= 1
        # B10 is idx 2, B11 is idx 3
        V[2, 6] += 1
        V[3, 6] += 1
        # Contributes to C00
        # C00 = k=0, i=0 -> idx 0
        W[0, 6] += 1
        
        return U, V, W

    def combine_decomps(decomp1, decomp2, n1, m1, p1, n2, m2, p2, offset_u, offset_v, offset_w):
        """
        Combines two decompositions into a larger one.
        decomp1: (U1, V1, W1) for n1xm1xp1
        decomp2: (U2, V2, W2) for n2xm2xp2
        offset_u, offset_v, offset_w: starting indices in the global matrices
        """
        U1, V1, W1 = decomp1
        U2, V2, W2 = decomp2
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        
        # Global dimensions
        global_n = n1 + n2 # This logic is tricky for non-additive splits, 
                           # but for our specific splits, we can handle them separately.
        
        # This generic combiner is hard because the splits are not always additive in all dimensions.
        # It's better to write specific recursive functions for the shapes we need.
        pass

    # Specific Recursive Functions
    
    def get_2_2_p(p):
        """
        Returns U, V, W for 2x2xp matrix multiplication.
        U: 4 x rank
        V: 2p x rank
        W: 2p x rank (wait, W is p x n = p x 2)
        """
        if p == 2:
            U, V, W = base_strassen()
            # U is 4x7 (2x2 flattened)
            # V is 4x7 (2x2 flattened)
            # W is 4x7 (2x2 flattened, k*2+i)
            return U, V, W
        
        # p is even
        p_half = p // 2
        # Split B into B1 (2 x p_half) and B2 (2 x p_half)
        # A is 2 x 2
        # AB = A B1 + A B2
        # We solve two 2x2xp_half problems.
        U1, V1, W1 = get_2_2_p(p_half)
        U2, V2, W2 = get_2_2_p(p_half)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        
        U = np.zeros((4, total_rank))
        V = np.zeros((2 * p, total_rank))
        W = np.zeros((2 * p, total_rank)) # W is p x 2 flattened
        
        # Copy U1 and U2
        U[:, :r1] = U1
        U[:, r1:] = U2
        
        # Copy V1 and V2
        # V1 corresponds to B1 (rows 0 to p_half-1)
        # V2 corresponds to B2 (rows p_half to 2*p_half-1)
        V[:p_half, :r1] = V1
        V[p_half:, r1:] = V2
        
        # Copy W1 and W2
        # W1 corresponds to C1 (rows 0 to p_half-1)
        # W2 corresponds to C2 (rows p_half to 2*p_half-1)
        W[:p_half, :r1] = W1
        W[p_half:, r1:] = W2
        
        return U, V, W

    def get_2_4_8():
        """
        Returns U, V, W for 2x4x8 matrix multiplication.
        U: 8 x rank (2x4 flattened)
        V: 32 x rank (4x8 flattened)
        W: 16 x rank (8x2 flattened)
        """
        # Split A into A1 (2x2) and A2 (2x2)
        # Split B into B1 (2x8) and B2 (2x8)
        # AB = A1 B1 + A2 B2
        # Solve two 2x2x8 problems.
        
        U1, V1, W1 = get_2_2_p(8)
        U2, V2, W2 = get_2_2_p(8)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        
        U = np.zeros((8, total_rank))
        V = np.zeros((32, total_rank))
        W = np.zeros((16, total_rank))
        
        # U1 is 4x7 (A1: rows 0,1 of A)
        # U2 is 4x7 (A2: rows 2,3 of A? No, A is 2x4.
        # A = [A1 A2] where A1 is 2x2, A2 is 2x2.
        # A1 occupies columns 0,1 of A. A2 occupies columns 2,3.
        # U1 corresponds to A1 elements.
        # U2 corresponds to A2 elements.
        
        # Flatten A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 for A1?
        # Wait, A is 2x4.
        # Indices:
        # 0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3)
        # 4: (1,0), 5: (1,1), 6: (1,2), 7: (1,3)
        
        # A1 is A[:, 0:2]. Elements: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5.
        # A2 is A[:, 2:4]. Elements: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7.
        
        # U1 is 4x7. Rows of U1 correspond to A1 local indices:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        # We need to map U1 rows to U global rows.
        # U1 row 0 -> U global row 0
        # U1 row 1 -> U global row 1
        # U1 row 2 -> U global row 4
        # U1 row 3 -> U global row 5
        
        # U2 is 4x7. Rows of U2 correspond to A2 local indices:
        # 0: (0,0) of A2 -> (0,2) of A -> Global 2
        # 1: (0,1) of A2 -> (0,3) of A -> Global 3
        # 2: (1,0) of A2 -> (1,2) of A -> Global 6
        # 3: (1,1) of A2 -> (1,3) of A -> Global 7
        
        map_u1 = [0, 1, 4, 5]
        map_u2 = [2, 3, 6, 7]
        
        for local_r, global_r in enumerate(map_u1):
            U[global_r, :r1] = U1[local_r, :]
        
        for local_r, global_r in enumerate(map_u2):
            U[global_r, r1:] = U2[local_r, :]
            
        # V1 is 16x7 (B1: 2x8 flattened).
        # B1 is B[0:2, :].
        # B1 local indices: (0,0)->0 ... (0,7)->7, (1,0)->8 ... (1,7)->15.
        # B1 global indices: same as local, since B1 is top half.
        # V2 is 16x7 (B2: 2x8 flattened).
        # B2 is B[2:4, :].
        # B2 local indices: (0,0)->0 ... (0,7)->7, (1,0)->8 ... (1,7)->15.
        # B2 global indices: (2,0)->16, (2,1)->17 ... (2,7)->23, (3,0)->24 ... (3,7)->31.
        
        # V1 rows 0..15 map to V global rows 0..15.
        # V2 rows 0..15 map to V global rows 16..31.
        
        V[:16, :r1] = V1
        V[16:, r1:] = V2
        
        # W1 is 16x7 (C1: 8x2 flattened).
        # C1 is C[:, 0:2]? No, C = A1 B1 + A2 B2.
        # C is 2x8.
        # C1 = A1 B1 is 2x8.
        # C2 = A2 B2 is 2x8.
        # C = C1 + C2.
        
        # W1 corresponds to C1.
        # W1 is 16x7. Flattened C1: (0,0)->0, (0,1)->1, (1,0)->2 ... (7,1)->15.
        # Wait, W is p x n flattened. p=8, n=2.
        # W index: k*n + i.
        # C1 is 2x8. Rows i=0,1. Cols k=0..7.
        # W1 rows correspond to C1 elements.
        # W1 row 0: C1[0,0] (i=0, k=0) -> Global C[0,0] -> W index 0*2+0=0.
        # W1 row 1: C1[0,1] (i=0, k=1) -> Global C[0,1] -> W index 1*2+0=2?
        # No, W index is k*n + i.
        # C1[0,0]: i=0, k=0 -> idx 0.
        # C1[0,1]: i=0, k=1 -> idx 1*2+0=2.
        # C1[1,0]: i=1, k=0 -> idx 0*2+1=1.
        # C1[1,1]: i=1, k=1 -> idx 1*2+1=3.
        
        # So W1 rows are not in the same order as W global rows.
        # W1 is generated by `get_2_2_p(8)`.
        # In `base_strassen`, W is 4x7.
        # W index: k*2+i.
        # W row 0: k=0, i=0.
        # W row 1: k=0, i=1.
        # W row 2: k=1, i=0.
        # W row 3: k=1, i=1.
        
        # In `get_2_2_p(p)`, W is 2p x rank.
        # The rows of W correspond to C elements.
        # C is 2xp.
        # W index: k*n + i.
        # For n=2, index = 2k + i.
        # Row 0 of W: k=0, i=0.
        # Row 1 of W: k=0, i=1.
        # Row 2 of W: k=1, i=0.
        # Row 3 of W: k=1, i=1.
        # ...
        # Row 2k: k, 0.
        # Row 2k+1: k, 1.
        
        # So W1 rows 0..15 correspond to C1 elements with k=0..7, i=0..1.
        # C1 is the top part of C? No, C1 is 2x8.
        # C1[i,k] = C[i,k].
        # So W1 row 2k corresponds to C[0,k] (if i=0) and C[1,k] (if i=1).
        # Wait, W row index is 2k+i.
        # So W1 row 2k is C[0,k].
        # W1 row 2k+1 is C[1,k].
        
        # W global has the same structure.
        # So W1 rows map directly to W global rows?
        # Yes, because C1 and C2 add up to C.
        # C1[i,k] is at W index 2k+i.
        # C2[i,k] is at W index 2k+i.
        # So we just add them.
        
        W[:, :r1] += W1
        W[:, r1:] += W2
        
        return U, V, W

    U, V, W = get_2_4_8()
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
