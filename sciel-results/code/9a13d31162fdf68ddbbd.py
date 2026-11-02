import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # We split the p=9 dimension into 4, 4, 1.
    # Part 1: 4x4x4 (Rank 49)
    # Part 2: 4x4x4 (Rank 49)
    # Part 3: 4x4x1 (Rank 16)
    
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_444 + rank_441
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Helper to build 2x2x2 Strassen decomposition
    def get_strassen_222():
        # Returns U, V, W for 2x2x2 with rank 7
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: i,j,k in {0,1}
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        
        U_222 = np.zeros((4, 7))
        V_222 = np.zeros((4, 7))
        W_222 = np.zeros((4, 7))
        
        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Term 1
        U_222[0, 0] = 1 # A11
        U_222[3, 0] = 1 # A22
        V_222[0, 0] = 1 # B11
        V_222[3, 0] = 1 # B22
        W_222[0, 0] = 1 # C11
        W_222[3, 0] = 1 # C22
        
        # Term 2
        U_222[2, 1] = 1 # A21
        U_222[3, 1] = 1 # A22
        V_222[0, 1] = 1 # B11
        W_222[2, 1] = 1 # C21
        W_222[3, 1] = -1 # C22
        
        # Term 3
        U_222[0, 2] = 1 # A11
        V_222[1, 2] = 1 # B12
        V_222[3, 2] = -1 # B22
        W_222[1, 2] = 1 # C12
        W_222[3, 2] = 1 # C22
        
        # Term 4
        U_222[3, 3] = 1 # A22
        V_222[2, 3] = 1 # B21
        V_222[0, 3] = -1 # B11
        W_222[0, 3] = 1 # C11
        W_222[2, 3] = 1 # C21
        
        # Term 5
        U_222[0, 4] = 1 # A11
        U_222[1, 4] = 1 # A12
        V_222[3, 4] = 1 # B22
        W_222[0, 4] = -1 # C11
        W_222[1, 4] = 1 # C12
        
        # Term 6
        U_222[2, 5] = 1 # A21
        U_222[0, 5] = -1 # A11
        V_222[0, 5] = 1 # B11
        V_222[1, 5] = 1 # B12
        W_222[3, 5] = 1 # C22
        
        # Term 7
        U_222[1, 6] = 1 # A12
        U_222[3, 6] = -1 # A22
        V_222[2, 6] = 1 # B21
        V_222[3, 6] = 1 # B22
        W_222[0, 6] = 1 # C11
        W_222[2, 6] = 1 # C21
        
        return U_222, V_222, W_222

    U_222, V_222, W_222 = get_strassen_222()
    
    # Build 4x4x4 decomposition
    # This uses 7 blocks of 2x2x2.
    # Each 2x2x2 block has rank 7. Total rank 49.
    # We need to map the 4x4x4 indices to the 2x2x2 blocks.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Strassen coefficients for 4x4x4 (same as 2x2x2)
    # A_s = sum alpha_ij A_ij
    # B_s = sum beta_ij B_ij
    # C_k = sum gamma_ks M_s
    
    # We iterate over the 7 Strassen terms s
    for s in range(7):
        # Get the 2x2x2 decomposition for this term
        # The 2x2x2 decomposition U_222[:, s] corresponds to the coefficients for A
        # But wait, the 2x2x2 decomposition IS the rank-7 decomposition.
        # So for the 4x4x4 case, we have 7 "super-terms".
        # Each super-term s is a product of a linear combination of A blocks and B blocks.
        # The result is a 2x2 matrix (block of C).
        # This result is then combined into C.
        
        # Actually, the recursive Strassen works like this:
        # The 4x4x4 tensor is a sum of 7 tensors, each of which is a 2x2x2 tensor
        # embedded in the 4x4x4 space.
        # The embedding is determined by the Strassen coefficients.
        
        # Let's define the Strassen coefficients for the 4x4x4 case.
        # They are the same as for 2x2x2.
        # M1 = (A11+A22)(B11+B22)
        # ...
        
        # For each s, we have a 2x2x2 multiplication.
        # The input A_s is a 2x2 matrix.
        # The input B_s is a 2x2 matrix.
        # The output M_s is a 2x2 matrix.
        
        # The 2x2x2 decomposition U_222, V_222, W_222 handles the internal 2x2x2 structure.
        # But we need to map the global 4x4 indices to the local 2x2 indices.
        
        # For term s:
        # A_s is formed from A blocks.
        # B_s is formed from B blocks.
        # M_s contributes to C blocks.
        
        # We need to know which A blocks and B blocks are used in term s.
        # And with what coefficients.
        
        # Let's hardcode the Strassen coefficients for the 7 terms.
        # a_coeffs[s][block]
        # b_coeffs[s][block]
        # c_coeffs[s][block]
        
        # Blocks are indexed 0,1,2,3 corresponding to 11, 12, 21, 22.
        
        a_coeffs = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0], # M6
            [0, 1, 0, -1]  # M7
        ]
        
        b_coeffs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1], # M3
            [0, 0, 1, -1], # M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        c_coeffs = [
            [1, 0, 0, 1], # M1 -> C11, C22
            [0, 0, 1, -1], # M2 -> C21, C22
            [0, 1, 0, 1], # M3 -> C12, C22
            [1, 0, 1, 0], # M4 -> C11, C21
            [-1, 1, 0, 0], # M5 -> C11, C12
            [0, 0, 0, 1], # M6 -> C22
            [1, 0, 1, 0]  # M7 -> C11, C21
        ]
        
        # Now, for each term s, we have a 2x2x2 multiplication.
        # The rank-7 decomposition of 2x2x2 is U_222, V_222, W_222.
        # We need to "inflate" this to 4x4x4.
        
        # The 2x2x2 indices are i', j', k' in {0,1}.
        # The 4x4x4 indices are i, j, k in {0,1,2,3}.
        # i = I * 2 + i'
        # j = J * 2 + j'
        # k = K * 2 + k'
        
        # For a given term s, the A blocks involved are those with a_coeffs[s][block] != 0.
        # The B blocks involved are those with b_coeffs[s][block] != 0.
        # The C blocks involved are those with c_coeffs[s][block] != 0.
        
        # For each of the 7 internal rank-1 terms of the 2x2x2 decomposition (let's call them t in 0..6):
        # We have U_222[i', j', t], V_222[j', k', t], W_222[k', i', t].
        # Wait, U_222 is 4x7, so U_222[i'*2+j', t].
        
        # The contribution to the global 4x4x4 tensor is:
        # For each internal term t:
        #   For each A block (I, J) with coeff alpha:
        #     For each B block (J, K) with coeff beta:
        #       For each C block (K, I) with coeff gamma:
        #         The rank-1 term is scaled by alpha * beta * gamma.
        #         The indices are mapped from (i', j', k') to (i, j, k).
        
        # This is getting complex. Let's simplify.
        # The 4x4x4 tensor is a sum of 7 terms, each of which is a 2x2x2 tensor.
        # The 2x2x2 tensor for term s is:
        # T_s = sum_{t=0}^6 u_t \otimes v_t \otimes w_t
        # Where u_t, v_t, w_t are 4-vectors (flattened 2x2).
        
        # The global tensor T is:
        # T = sum_{s=0}^6 sum_{t=0}^6 (alpha_s \otimes beta_s \otimes gamma_s) * (u_t \otimes v_t \otimes w_t)
        
        # This means we have 49 rank-1 terms in total.
        
        for t in range(7):
            # Get the internal rank-1 vectors
            u_t = U_222[:, t]
            v_t = V_222[:, t]
            w_t = W_222[:, t]
            
            # For each A block (I, J)
            for I in range(2):
                for J in range(2):
                    block_a = I * 2 + J
                    alpha = a_coeffs[s][block_a]
                    if alpha == 0: continue
                    
                    # For each B block (J, K)
                    for K in range(2):
                        block_b = J * 2 + K
                        beta = b_coeffs[s][block_b]
                        if beta == 0: continue
                        
                        # For each C block (K, I)
                        for I_c in range(2):
                            for J_c in range(2):
                                block_c = I_c * 2 + J_c
                                gamma = c_coeffs[s][block_c]
                                if gamma == 0: continue
                                
                                # The contribution is alpha * beta * gamma
                                scale = alpha * beta * gamma
                                if scale == 0: continue
                                
                                # The indices in the 4x4x4 space
                                # i = I * 2 + i'
                                # j = J * 2 + j'
                                # k = K * 2 + k'
                                
                                # u_t is indexed by i', j'.
                                # v_t is indexed by j', k'.
                                # w_t is indexed by k', i'.
                                
                                for i_prime in range(2):
                                    for j_prime in range(2):
                                        u_val = u_t[i_prime * 2 + j_prime]
                                        if u_val == 0: continue
                                        
                                        i = I * 2 + i_prime
                                        j = J * 2 + j_prime
                                        
                                        for k_prime in range(2):
                                            v_val = v_t[j_prime * 2 + k_prime]
                                            if v_val == 0: continue
                                            
                                            k = K * 2 + k_prime
                                            
                                            w_val = w_t[k_prime * 2 + i_prime]
                                            if w_val == 0: continue
                                            
                                            # The rank-1 term index is s * 7 + t
                                            r = s * 7 + t
                                            
                                            # Add to U, V, W
                                            U_444[i * 4 + j, r] += scale * u_val
                                            V_444[j * 4 + k, r] += scale * v_val
                                            W_444[k * 4 + i, r] += scale * w_val
                                            
    # This loop is very slow. 7 * 7 * 4 * 4 * 4 * 4 * 2 * 2 * 2 = 7*7*4*4*4*4*8 = 235928 iterations.
    # This is fine for Python if optimized, but let's check.
    # Actually, the inner loops are small.
    
    # Now fill the main U, V, W matrices.
    
    # Part 1: 4x4x4 (cols 0-48)
    U[:, 0:49] = U_444
    # V for Part 1: B1 is 4x4, cols 0-3 of B.
    # V_444 is 16x49.
    # V is 36x114.
    # Rows 0-15 of V correspond to j=0..3, k=0..3.
    V[0:16, 0:49] = V_444
    # W for Part 1: C1 is 4x4, cols 0-3 of C.
    # W_444 is 16x49.
    # W is 36x114.
    # Rows 0-15 of W correspond to k=0..3, i=0..3.
    W[0:16, 0:49] = W_444
    
    # Part 2: 4x4x4 (cols 49-97)
    U[:, 49:98] = U_444
    # V for Part 2: B2 is 4x4, cols 4-7 of B.
    # Rows 16-31 of V correspond to j=0..3, k=4..7.
    # We need to shift the k index by 4.
    # V_444[j*4+k, r] -> V[j*4+(k+4), r]
    V[16:32, 49:98] = V_444
    # W for Part 2: C2 is 4x4, cols 4-7 of C.
    # Rows 16-31 of W correspond to k=4..7, i=0..3.
    W[16:32, 49:98] = W_444
    
    # Part 3: 4x4x1 (cols 98-113)
    # Schoolbook decomposition for 4x4x1.
    # Rank 16.
    # U is 16x16, V is 4x16, W is 4x16.
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((4, 16))
    W_441 = np.zeros((4, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i * 4 + j, col] = 1
            V_441[j, col] = 1
            W_441[i, col] = 1
            col += 1
            
    U[:, 98:114] = U_441
    # V for Part 3: B3 is 4x1, col 8 of B.
    # Rows 32-35 of V correspond to j=0..3, k=8.
    V[32:36, 98:114] = V_441
    # W for Part 3: C3 is 4x1, col 8 of C.
    # Rows 32-35 of W correspond to k=8, i=0..3.
    W[32:36, 98:114] = W_441
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
