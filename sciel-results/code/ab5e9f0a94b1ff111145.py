import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We construct two 4x4x4 decompositions and combine them.
    # Rank 49 for each. Total 98.
    
    def get_4x4x4_factors():
        # Returns U, V, W for 4x4x4 multiplication
        # U: 16x49, V: 16x49, W: 16x49
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # Inner Strassen factors for 2x2
        # U_inner, V_inner, W_inner for 2x2x2
        # Indices for U_inner: a,b,c,d (0,1,2,3)
        # Indices for V_inner: e,f,g,h (0,1,2,3)
        # Indices for W_inner: C00, C10, C01, C11 (0,1,2,3) -> k*2+i
        # C00: k=0,i=0 -> 0
        # C10: k=0,i=1 -> 1
        # C01: k=1,i=0 -> 2
        # C11: k=1,i=1 -> 3
        
        # Strassen 2x2 terms:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Coefficients for A (U) and B (V) for each M
        # U is 4x7, V is 4x7
        U_in = np.zeros((4, 7))
        V_in = np.zeros((4, 7))
        
        # M1
        U_in[0, 0] = 1; U_in[3, 0] = 1 # a+d
        V_in[0, 0] = 1; V_in[3, 0] = 1 # e+h
        # M2
        U_in[2, 1] = 1; U_in[3, 1] = 1 # c+d
        V_in[0, 1] = 1 # e
        # M3
        U_in[0, 2] = 1 # a
        V_in[1, 2] = 1; V_in[3, 2] = -1 # f-h
        # M4
        U_in[3, 3] = 1 # d
        V_in[2, 3] = 1; V_in[0, 3] = -1 # g-e
        # M5
        U_in[0, 4] = 1; U_in[1, 4] = 1 # a+b
        V_in[3, 4] = 1 # h
        # M6
        U_in[2, 5] = 1; U_in[0, 5] = -1 # c-a
        V_in[0, 5] = 1; V_in[1, 5] = 1 # e+f
        # M7
        U_in[1, 6] = 1; U_in[3, 6] = -1 # b-d
        V_in[2, 6] = 1; V_in[3, 6] = 1 # g+h
        
        # W_in is 4x7
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        W_in = np.zeros((4, 7))
        W_in[0, 0] = 1 # C00 M1
        W_in[0, 3] = 1 # C00 M4
        W_in[0, 4] = -1 # C00 M5
        W_in[0, 6] = 1 # C00 M7
        
        W_in[1, 1] = 1 # C10 M2
        W_in[1, 3] = 1 # C10 M4
        
        W_in[2, 2] = 1 # C01 M3
        W_in[2, 4] = 1 # C01 M5
        
        W_in[3, 0] = 1 # C11 M1
        W_in[3, 1] = -1 # C11 M2
        W_in[3, 2] = 1 # C11 M3
        W_in[3, 5] = 1 # C11 M6
        
        # Outer Strassen terms
        # A blocks: 00(0-3), 01(4-7), 10(8-11), 11(12-15)
        # B blocks: 00(0-3), 01(4-7), 10(8-11), 11(12-15)
        # C blocks: 00, 01, 10, 11
        
        # Outer M definitions:
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # M2 = (A10+A11)B00 -> C10, C11
        # M3 = A00(B01-B11) -> C01, C11
        # M4 = A11(B10-B00) -> C00, C10
        # M5 = (A00+A01)B11 -> C00, C01
        # M6 = (A10-A00)(B00+B01) -> C11
        # M7 = (A01-A11)(B10+B11) -> C00
        
        # For each outer term r, we have linear combos of blocks for X and Y.
        # X = sum coeff_block * A_block
        # Y = sum coeff_block * B_block
        # The inner decomposition of X Y is sum_s (U_in_s X) (V_in_s Y) W_in_s
        # So for each r, s:
        # U_full = sum_block coeff_X_block * (U_in_s placed in block)
        # V_full = sum_block coeff_Y_block * (V_in_s placed in block)
        # W_full = coeff_W_block * (W_in_s placed in block)
        
        # Define outer coefficients
        # r: (X_coeffs, Y_coeffs, W_block, W_coeff)
        # X_coeffs: dict block_index -> coeff
        # Y_coeffs: dict block_index -> coeff
        
        outer_terms = [
            # M1
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # C00:1, C11:1
            # M2
            ({2:1, 3:1}, {0:1}, {1:1, 3:-1}), # C10:1, C11:-1
            # M3
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}), # C01:1, C11:1
            # M4
            ({3:1}, {2:1, 0:-1}, {0:1, 1:1}), # C00:1, C10:1
            # M5
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}), # C00:-1, C01:1
            # M6
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}), # C11:1
            # M7
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1}), # C00:1
        ]
        
        # Block offsets for U (A)
        # A00: 0, A01: 4, A10: 8, A11: 12
        A_offsets = [0, 4, 8, 12]
        # Block offsets for V (B)
        B_offsets = [0, 4, 8, 12]
        # Block offsets for W (C)
        # C00: 0,1,4,5
        # C01: 8,9,12,13
        # C10: 2,3,6,7
        # C11: 10,11,14,15
        # We need to map local W indices (0,1,2,3) to global indices for each block.
        # Local 0 (C00) -> Global 0
        # Local 1 (C10) -> Global 1
        # Local 2 (C01) -> Global 4
        # Local 3 (C11) -> Global 5
        # Wait, local indices are for the 2x2 matrix.
        # Local 0: i=0, k=0. Global: I=0, K=0 -> 0*4+0=0.
        # Local 1: i=1, k=0. Global: I=1, K=0 -> 0*4+1=1.
        # Local 2: i=0, k=1. Global: I=0, K=1 -> 1*4+0=4.
        # Local 3: i=1, k=1. Global: I=1, K=1 -> 1*4+1=5.
        
        # For C00 block (Rb=0, Cb=0):
        # Local 0 -> 0
        # Local 1 -> 1
        # Local 2 -> 4
        # Local 3 -> 5
        W_map_00 = [0, 1, 4, 5]
        
        # For C01 block (Rb=0, Cb=1):
        # I = i, K = k+2.
        # Local 0 (i=0,k=0) -> I=0, K=2 -> 2*4+0=8.
        # Local 1 (i=1,k=0) -> I=1, K=2 -> 2*4+1=9.
        # Local 2 (i=0,k=1) -> I=0, K=3 -> 3*4+0=12.
        # Local 3 (i=1,k=1) -> I=1, K=3 -> 3*4+1=13.
        W_map_01 = [8, 9, 12, 13]
        
        # For C10 block (Rb=1, Cb=0):
        # I = i+2, K = k.
        # Local 0 (i=0,k=0) -> I=2, K=0 -> 0*4+2=2.
        # Local 1 (i=1,k=0) -> I=3, K=0 -> 0*4+3=3.
        # Local 2 (i=0,k=1) -> I=2, K=1 -> 1*4+2=6.
        # Local 3 (i=1,k=1) -> I=3, K=1 -> 1*4+3=7.
        W_map_10 = [2, 3, 6, 7]
        
        # For C11 block (Rb=1, Cb=1):
        # I = i+2, K = k+2.
        # Local 0 -> I=2, K=2 -> 2*4+2=10.
        # Local 1 -> I=3, K=2 -> 2*4+3=11.
        # Local 2 -> I=2, K=3 -> 3*4+2=14.
        # Local 3 -> I=3, K=3 -> 3*4+3=15.
        W_map_11 = [10, 11, 14, 15]
        
        W_maps = {0: W_map_00, 1: W_map_01, 2: W_map_10, 3: W_map_11}
        # Note: In outer_terms, W_block keys are 0,1,2,3 corresponding to C00, C01, C10, C11?
        # My outer_terms list used:
        # M1: {0:1, 3:1} -> C00, C11.
        # M2: {1:1, 3:-1} -> C10, C11.
        # M3: {1:1, 3:1} -> C01, C11.
        # M4: {0:1, 1:1} -> C00, C10.
        # M5: {0:-1, 1:1} -> C00, C01.
        # M6: {3:1} -> C11.
        # M7: {0:1} -> C00.
        
        # So keys: 0=C00, 1=C01, 2=C10, 3=C11?
        # Let's check M2: C10 and C11.
        # In my W_maps, 1 is C01, 2 is C10.
        # So in outer_terms, I should use key 2 for C10 and key 3 for C11.
        # Let's re-verify outer_terms keys.
        # M1: C00 (0), C11 (3). Correct.
        # M2: C10 (2), C11 (3). My code had {1:1, 3:-1}. This is WRONG. 1 is C01.
        # I need to fix the keys in outer_terms.
        
        corrected_outer_terms = [
            # M1
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # C00, C11
            # M2
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}), # C10, C11
            # M3
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}), # C01, C11
            # M4
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}), # C00, C10
            # M5
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}), # C00, C01
            # M6
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}), # C11
            # M7
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1}), # C00
        ]
        
        for r_idx, (x_coeffs, y_coeffs, w_coeffs) in enumerate(corrected_outer_terms):
            for s_idx in range(7):
                col = r_idx * 7 + s_idx
                
                # U construction
                # U_in[s_idx] is vector of 4.
                # For each block in x_coeffs:
                for b_idx, coeff in x_coeffs.items():
                    offset = A_offsets[b_idx]
                    U[offset:offset+4, col] += coeff * U_in[:, s_idx]
                
                # V construction
                for b_idx, coeff in y_coeffs.items():
                    offset = B_offsets[b_idx]
                    V[offset:offset+4, col] += coeff * V_in[:, s_idx]
                
                # W construction
                for b_idx, coeff in w_coeffs.items():
                    w_map = W_maps[b_idx]
                    for l_idx in range(4):
                        W[w_map[l_idx], col] += coeff * W_in[l_idx, s_idx]
                        
        return U, V, W

    U_444, V_444, W_444 = get_4x4x4_factors()
    
    # Now combine two 4x4x4 decompositions for 4x4x8
    # Rank 98
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank)) # m*p = 4*8 = 32
    W = np.zeros((32, rank)) # p*n = 8*4 = 32
    
    # First 4 columns of output (k=0..3)
    # A is 4x4. B_left is 4x4 (cols 0..3 of B).
    # U is same.
    # V: B_left indices 0..15 (j*4+k, j=0..3, k=0..3).
    # W: C_left indices k*4+i, k=0..3, i=0..3.
    
    U[:, 0:49] = U_444
    V[0:16, 0:49] = V_444
    W[0:16, 0:49] = W_444
    
    # Last 4 columns of output (k=4..7)
    # B_right is 4x4 (cols 4..7 of B).
    # V indices for B_right: j*4 + (k+4) = j*4 + k + 4.
    # So V[4:20, 49:98] = V_444?
    # B index is j*p+k. p=8.
    # B_right element (j, k') where k'=0..3 corresponds to global k=k'+4.
    # Index = j*8 + (k'+4).
    # V_444 index is j*4 + k'.
    # So we need to map V_444 indices to V indices.
    
    for j in range(4):
        for k_prime in range(4):
            v_444_idx = j * 4 + k_prime
            v_global_idx = j * 8 + (k_prime + 4)
            V[v_global_idx, 49:98] = V_444[v_444_idx, :]
            
    # W for C_right: k=4..7.
    # W index = k*4 + i.
    # k = k_prime + 4.
    # Index = (k_prime+4)*4 + i = k_prime*4 + 16 + i.
    # W_444 index = k_prime*4 + i.
    
    for k_prime in range(4):
        for i in range(4):
            w_444_idx = k_prime * 4 + i
            w_global_idx = (k_prime + 4) * 4 + i
            W[w_global_idx, 49:98] = W_444[w_444_idx, :]
            
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
