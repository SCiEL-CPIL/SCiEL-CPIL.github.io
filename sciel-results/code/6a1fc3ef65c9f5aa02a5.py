import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 decomposition using recursive Strassen (Rank 49)
    # We generate the decomposition for 2x2x2 first
    def get_2x2_decomp():
        # U, V, W for 2x2x2 Strassen
        # A indices: 0,1,2,3 (00, 01, 10, 11)
        # B indices: 0,1,2,3
        # C indices: 0,1,2,3
        U = []
        V = []
        W = []
        
        # M1 = (A00+A11)(B00+B11)
        U.append([1, 0, 0, 1])
        V.append([1, 0, 0, 1])
        W.append([1, 0, 0, 1]) # C00 += 1, C11 += 1
        
        # M2 = (A10+A11)B00
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        W.append([0, 0, 1, -1]) # C10 += 1, C11 -= 1
        
        # M3 = A00(B01-B11)
        U.append([1, 0, 0, 0])
        V.append([0, 1, 0, -1])
        W.append([0, 1, 0, 1]) # C01 += 1, C11 += 1
        
        # M4 = A11(B10-B00)
        U.append([0, 0, 0, 1])
        V.append([0, 0, 1, -1])
        W.append([-1, 0, 1, 0]) # C00 -= 1, C10 += 1
        
        # M5 = (A00+A01)B11
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        W.append([1, 0, 0, 0]) # C00 += 1
        
        # M6 = (A10-A00)(B00+B01)
        U.append([0, 0, 1, -1])
        V.append([1, 1, 0, 0])
        W.append([0, 0, 0, 1]) # C11 += 1
        
        # M7 = (A01-A11)(B10+B11)
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        W.append([-1, 1, 0, 0]) # C00 -= 1, C01 += 1
        
        return np.array(U).T, np.array(V).T, np.array(W).T

    U2, V2, W2 = get_2x2_decomp()
    
    # Now construct 4x4x4 decomposition
    # A is 4x4, blocks A00, A01, A10, A11 (each 2x2)
    # Global indices for A:
    # 0,1,4,5 (A00)
    # 2,3,6,7 (A01)
    # 8,9,12,13 (A10)
    # 10,11,14,15 (A11)
    
    # We define the 7 Strassen linear combinations for 4x4
    # Each combination produces a 2x2 matrix A' and B'
    # We then decompose A'B' using the 2x2 decomp
    
    U4 = []
    V4 = []
    W4 = []
    
    # Helper to map local 2x2 indices to global 4x4 indices
    # Block (br, bc) has global row offset br*2 and col offset bc*2
    # Local index (lr, lc) maps to global (br*2+lr, bc*2+lc)
    # Global index in vector form: row*4 + col
    
    def get_block_indices(br, bc):
        # Returns list of 4 global indices for the 2x2 block
        return [ (br*2 + lr)*4 + (bc*2 + lc) for lr in range(2) for lc in range(2) ]
    
    # Strassen terms for 4x4
    # k=0: M1 = (A00+A11)(B00+B11)
    # k=1: M2 = (A10+A11)B00
    # k=2: M3 = A00(B01-B11)
    # k=3: M4 = A11(B10-B00)
    # k=4: M5 = (A00+A01)B11
    # k=5: M6 = (A10-A00)(B00+B01)
    # k=6: M7 = (A01-A11)(B10+B11)
    
    # For each k, we define the linear combinations for A' and B'
    # A' is a 2x2 matrix. Its entries are linear combinations of A's entries.
    # We represent A' as a 4-vector (flattened 2x2).
    # Similarly for B'.
    
    # We need to express each entry of A' as a linear combination of global A entries.
    # Let L_A[k] be a 4x16 matrix such that A'_k = L_A[k] @ A_flat
    # Similarly L_B[k]
    
    # And we need to map the resulting C' (2x2) to the global C.
    # Let R_C[k] be a 4x16 matrix such that C_flat += R_C[k] @ C'_flat
    
    # Actually, it's easier to just iterate through the 7 terms of the 2x2 decomp
    # for each of the 7 Strassen terms.
    
    # Precompute the linear combinations for A and B for each Strassen term
    # A' entries:
    # M1: A00+A11. A' = A00+A11.
    #    A'_00 = A_00 + A_11 (local 0,0 of A00 + local 0,0 of A11)
    #    This is getting tedious.
    
    # Alternative: Just use the fact that Strassen is linear.
    # The 49 multiplications are of the form:
    # (u_i . A) * (v_i . B) * w_i
    # where u_i, v_i, w_i are vectors of length 16.
    
    # I will generate these 49 vectors by combining the 2x2 decomp with the 4x4 Strassen structure.
    
    # For each Strassen term k (0..6):
    #   For each 2x2 decomp term j (0..6):
    #       The multiplication is m_kj = (u2_j . A'_k) * (v2_j . B'_k)
    #       u2_j is a 4-vector (for 2x2 A').
    #       A'_k is a linear combination of A blocks.
    #       So u2_j . A'_k is a linear combination of A entries.
    #       Let U_kj be the resulting 16-vector.
    #       Similarly V_kj.
    #       The contribution to C is w2_j (4-vector for 2x2 C') mapped to global C.
    #       Let W_kj be the resulting 16-vector.
    #       Note: The global Strassen term k also has a coefficient/sign in the final C.
    #       This is handled by the W2 mapping? No.
    #       The W2 vectors from get_2x2_decomp are for the local C'.
    #       The global C is a sum of the local C' placed in blocks, with signs from the 4x4 Strassen formula.
    #       Wait, the 4x4 Strassen formula is:
    #       C00 = M1 - M4 + M5 - M7
    #       C01 = M3 + M7
    #       C10 = M2 + M4
    #       C11 = M1 - M2 + M3 + M6
    #       Here M_k are 2x2 matrices.
    #       So for each k, the local C' (which is M_k) is added to the global C blocks with specific signs.
    #       So W_kj should be the W2_j vector, but placed in the correct global block and scaled by the sign.
    
    # Let's define the block mappings and signs.
    # C00 is block (0,0). C01 is (0,1). C10 is (1,0). C11 is (1,1).
    
    # Signs for M_k in each C block:
    # M1: C00(+), C11(+)
    # M2: C10(+), C11(-)
    # M3: C01(+), C11(+)
    # M4: C00(-), C10(+)
    # M5: C00(+)
    # M6: C11(+)
    # M7: C00(-), C01(+)
    
    signs = {
        0: [(0,0,1), (1,1,1)],   # M1
        1: [(1,0,1), (1,1,-1)],  # M2
        2: [(0,1,1), (1,1,1)],   # M3
        3: [(0,0,-1), (1,0,1)],  # M4
        4: [(0,0,1)],            # M5
        5: [(1,1,1)],            # M6
        6: [(0,0,-1), (0,1,1)]   # M7
    }
    
    # Linear combinations for A' and B' for each M_k
    # A' is 2x2. Entries are sums of A blocks.
    # We define A'_k as a 4x16 matrix.
    
    def make_A_comb(k):
        # Returns 4x16 matrix
        M = np.zeros((4, 16))
        # Block indices:
        # A00: 0,1,4,5
        # A01: 2,3,6,7
        # A10: 8,9,12,13
        # A11: 10,11,14,15
        
        if k == 0: # A00+A11
            M[0, 0] = 1; M[0, 10] = 1
            M[1, 1] = 1; M[1, 11] = 1
            M[2, 4] = 1; M[2, 14] = 1
            M[3, 5] = 1; M[3, 15] = 1
        elif k == 1: # A10+A11
            M[0, 8] = 1; M[0, 10] = 1
            M[1, 9] = 1; M[1, 11] = 1
            M[2, 12] = 1; M[2, 14] = 1
            M[3, 13] = 1; M[3, 15] = 1
        elif k == 2: # A00
            M[0, 0] = 1
            M[1, 1] = 1
            M[2, 4] = 1
            M[3, 5] = 1
        elif k == 3: # A11
            M[0, 10] = 1
            M[1, 11] = 1
            M[2, 14] = 1
            M[3, 15] = 1
        elif k == 4: # A00+A01
            M[0, 0] = 1; M[0, 2] = 1
            M[1, 1] = 1; M[1, 3] = 1
            M[2, 4] = 1; M[2, 6] = 1
            M[3, 5] = 1; M[3, 7] = 1
        elif k == 5: # A10-A00
            M[0, 8] = 1; M[0, 0] = -1
            M[1, 9] = 1; M[1, 1] = -1
            M[2, 12] = 1; M[2, 4] = -1
            M[3, 13] = 1; M[3, 5] = -1
        elif k == 6: # A01-A11
            M[0, 2] = 1; M[0, 10] = -1
            M[1, 3] = 1; M[1, 11] = -1
            M[2, 6] = 1; M[2, 14] = -1
            M[3, 7] = 1; M[3, 15] = -1
        return M

    def make_B_comb(k):
        M = np.zeros((4, 16))
        if k == 0: # B00+B11
            M[0, 0] = 1; M[0, 10] = 1
            M[1, 1] = 1; M[1, 11] = 1
            M[2, 4] = 1; M[2, 14] = 1
            M[3, 5] = 1; M[3, 15] = 1
        elif k == 1: # B00
            M[0, 0] = 1
            M[1, 1] = 1
            M[2, 4] = 1
            M[3, 5] = 1
        elif k == 2: # B01-B11
            M[0, 2] = 1; M[0, 10] = -1
            M[1, 3] = 1; M[1, 11] = -1
            M[2, 6] = 1; M[2, 14] = -1
            M[3, 7] = 1; M[3, 15] = -1
        elif k == 3: # B10-B00
            M[0, 8] = 1; M[0, 0] = -1
            M[1, 9] = 1; M[1, 1] = -1
            M[2, 12] = 1; M[2, 4] = -1
            M[3, 13] = 1; M[3, 5] = -1
        elif k == 4: # B11
            M[0, 10] = 1
            M[1, 11] = 1
            M[2, 14] = 1
            M[3, 15] = 1
        elif k == 5: # B00+B01
            M[0, 0] = 1; M[0, 2] = 1
            M[1, 1] = 1; M[1, 3] = 1
            M[2, 4] = 1; M[2, 6] = 1
            M[3, 5] = 1; M[3, 7] = 1
        elif k == 6: # B10+B11
            M[0, 8] = 1; M[0, 10] = 1
            M[1, 9] = 1; M[1, 11] = 1
            M[2, 12] = 1; M[2, 14] = 1
            M[3, 13] = 1; M[3, 15] = 1
        return M

    for k in range(7):
        LA = make_A_comb(k)
        LB = make_B_comb(k)
        
        for j in range(7):
            # u2_j is column j of U2 (4x1)
            # v2_j is column j of V2 (4x1)
            # w2_j is column j of W2 (4x1)
            
            u_vec_2 = U2[:, j]
            v_vec_2 = V2[:, j]
            w_vec_2 = W2[:, j]
            
            # U_kj = LA @ u_vec_2 (16x1)
            U_kj = LA @ u_vec_2
            V_kj = LB @ v_vec_2
            
            # W_kj: Map w_vec_2 to global C
            # w_vec_2 is 4x1, corresponding to C' (2x2)
            # C' entries: 0,1,2,3 -> C'00, C'01, C'10, C'11
            # We need to add this to global C blocks with signs from signs[k]
            
            W_kj = np.zeros(16)
            for (br, bc, sign) in signs[k]:
                # Map local C' indices to global C indices
                # Local 0 -> Global (br*2, bc*2)
                # Local 1 -> Global (br*2, bc*2+1)
                # Local 2 -> Global (br*2+1, bc*2)
                # Local 3 -> Global (br*2+1, bc*2+1)
                
                g0 = (br*2)*4 + (bc*2)
                g1 = (br*2)*4 + (bc*2+1)
                g2 = (br*2+1)*4 + (bc*2)
                g3 = (br*2+1)*4 + (bc*2+1)
                
                W_kj[g0] += sign * w_vec_2[0]
                W_kj[g1] += sign * w_vec_2[1]
                W_kj[g2] += sign * w_vec_2[2]
                W_kj[g3] += sign * w_vec_2[3]
            
            U4.append(U_kj)
            V4.append(V_kj)
            W4.append(W_kj)
            
    U4 = np.array(U4).T
    V4 = np.array(V4).T
    W4 = np.array(W4).T
    
    # Part 2: 4x4x1 decomposition (Rank 4)
    # C_col = A * B_col
    # B_col is 4x1.
    # This is 4 dot products.
    # U_part2: 16x4
    # V_part2: 20x4 (only last column of B, indices 16,17,18,19)
    # W_part2: 20x4 (last column of C, indices 16,17,18,19)
    
    U5 = np.zeros((16, 4))
    V5 = np.zeros((20, 4))
    W5 = np.zeros((20, 4))
    
    for i in range(4):
        # C_i4 = sum_j A_ij B_j4
        # Term i:
        # U: 1 at A_i0, A_i1, A_i2, A_i3
        # V: 1 at B_04, B_14, B_24, B_34
        # W: 1 at C_i4
        
        U5[i*4, i] = 1
        U5[i*4+1, i] = 1
        U5[i*4+2, i] = 1
        U5[i*4+3, i] = 1
        
        V5[16, i] = 1
        V5[17, i] = 1
        V5[18, i] = 1
        V5[19, i] = 1
        
        W5[16+i, i] = 1
        
    # Combine
    U = np.hstack([U4, U5])
    V = np.hstack([V4, V5])
    W = np.hstack([W4, W5])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
