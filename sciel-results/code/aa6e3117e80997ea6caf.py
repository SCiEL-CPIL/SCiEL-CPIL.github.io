import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # We will use a Strassen-based decomposition.
    # Strategy:
    # Pad the 2x4 * 4x6 problem to a 4x4 * 4x8 problem? No, that's wasteful.
    # Instead, we can view the 2x4 * 4x6 multiplication as two 2x4 * 4x3 multiplications.
    # Or, better:
    # Split A (2x4) into A1 (2x2) and A2 (2x2) horizontally.
    # Split B (4x6) into B1 (4x3) and B2 (4x3) horizontally.
    # C = A1*B1_top + A2*B1_bottom + ... wait.
    
    # Let's use the standard blocking:
    # A is 2x4. B is 4x6.
    # Split the inner dimension (size 4) into two parts of size 2.
    # A = [A_left (2x2), A_right (2x2)]
    # B = [B_top (2x6); B_bottom (2x6)]
    # C = A_left * B_top + A_right * B_bottom
    
    # Now we have two sub-problems:
    # 1. 2x2 * 2x6
    # 2. 2x2 * 2x6
    # Since they share the same A structure but different B, we can't simply add ranks unless we treat them independently.
    # However, we can solve 2x2 * 2x6 efficiently.
    
    # How to solve 2x2 * 2x6?
    # Split B (2x6) into three 2x2 blocks: B1, B2, B3.
    # C = A*B1 + A*B2 + A*B3.
    # This is 3 separate 2x2 * 2x2 multiplications.
    # We can use Strassen's algorithm for each 2x2 * 2x2, which has rank 7.
    # So 2x2 * 2x6 can be done with rank 7 + 7 + 7 = 21?
    # Wait, the A matrix is the SAME for all three blocks.
    # Can we share the Strassen intermediates?
    
    # Strassen computes:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B10)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    
    # For B1, B2, B3, we have:
    # B1 = [[b1_00, b1_01], [b1_10, b1_11]]
    # B2 = [[b2_00, b2_01], [b2_10, b2_11]]
    # B3 = [[b3_00, b3_01], [b3_10, b3_11]]
    
    # The A-related factors in Strassen are:
    # u1 = A00+A11
    # u2 = A10+A11
    # u3 = A00
    # u4 = A11
    # u5 = A00+A01
    # u6 = A10-A00
    # u7 = A01-A11
    
    # These u's are fixed for the A matrix.
    # The B-related factors are linear combinations of B entries.
    # For B1:
    # v1_1 = b1_00 + b1_11
    # v2_1 = b1_00
    # v3_1 = b1_10 - b1_11
    # v4_1 = b1_01 - b1_10
    # v5_1 = b1_11
    # v6_1 = b1_00 + b1_10
    # v7_1 = b1_01 + b1_11
    
    # The products are u_i * v_j_i.
    # Since u_i are constant, we can define the U vectors for the rank-1 terms based on u_i.
    # The V vectors will depend on the specific B block.
    
    # Let's construct the decomposition for 2x2 * 2x6.
    # Rank = 21 (7 per 2x2 block).
    
    def build_2x2x2x6_decomp():
        """
        Decomposition for 2x2 * 2x6.
        A is 2x2 (indices 0..3), B is 2x6 (indices 0..11), C is 2x6 (indices 0..11).
        B is split into B1 (cols 0-3), B2 (cols 4-7), B3 (cols 8-11).
        C is split into C1 (cols 0-3), C2 (cols 4-7), C3 (cols 8-11).
        """
        U_sub_dim = 4
        V_sub_dim = 12
        W_sub_dim = 12
        rank_sub = 21
        
        U_sub = np.zeros((U_sub_dim, rank_sub), dtype=np.int32)
        V_sub = np.zeros((V_sub_dim, rank_sub), dtype=np.int32)
        W_sub = np.zeros((W_sub_dim, rank_sub), dtype=np.int32)
        
        # Strassen A-factors (u1..u7)
        # u1 = A00+A11 -> U indices 0, 3
        # u2 = A10+A11 -> U indices 2, 3
        # u3 = A00     -> U index 0
        # u4 = A11     -> U index 3
        # u5 = A00+A01 -> U indices 0, 1
        # u6 = A10-A00 -> U indices 2, 0 (-1)
        # u7 = A01-A11 -> U indices 1, 3 (-1)
        
        # We need to map these to the global U vector.
        # But first, let's define the local U vectors for each term.
        # Term t corresponds to u_i * v_j.
        # U_sub[:, t] is the vector u_i.
        
        # Let's define the 7 u-vectors:
        u_vecs = []
        u1 = np.array([1, 0, 0, 1]); u_vecs.append(u1)
        u2 = np.array([0, 0, 1, 1]); u_vecs.append(u2)
        u3 = np.array([1, 0, 0, 0]); u_vecs.append(u3)
        u4 = np.array([0, 0, 0, 1]); u_vecs.append(u4)
        u5 = np.array([1, 1, 0, 0]); u_vecs.append(u5)
        u6 = np.array([-1, 0, 1, 0]); u_vecs.append(u6)
        u7 = np.array([0, 1, 0, -1]); u_vecs.append(u7)
        
        # Now for each B block (1, 2, 3), we define the 7 v-vectors and w-vectors.
        # B block 1: indices 0..3 in V, 0..3 in W
        # B block 2: indices 4..7 in V, 4..7 in W
        # B block 3: indices 8..11 in V, 8..11 in W
        
        # Vectors for B block s (offset s*4):
        # v1 = B00+B11 -> indices s*4+0, s*4+3
        # v2 = B00     -> index s*4+0
        # v3 = B10-B11 -> indices s*4+1, s*4+3 (-1)
        # v4 = B01-B10 -> indices s*4+2, s*4+1 (-1)
        # v5 = B11     -> index s*4+3
        # v6 = B00+B10 -> indices s*4+0, s*4+1
        # v7 = B01+B11 -> indices s*4+2, s*4+3
        
        # W vectors for C block s (offset s*4):
        # C00 = m1+m4-m5+m7
        # C01 = m3+m5
        # C10 = m2+m4
        # C11 = m1-m2+m3+m6
        # W indices: s*4+0 (C00), s*4+1 (C10), s*4+2 (C01), s*4+3 (C11)
        # Wait, C is 2x6. C[i, k].
        # For block s, C columns are 2s, 2s+1.
        # C00 is C[0, 2s], C01 is C[0, 2s+1], C10 is C[1, 2s], C11 is C[1, 2s+1].
        # So W indices:
        # C00 -> 2s
        # C10 -> 2s+1
        # C01 -> 2s+2
        # C11 -> 2s+3
        
        # Let's re-verify W indices for 2x2 * 2x2.
        # C = [[C00, C01], [C10, C11]].
        # Index c = k*2+i.
        # C00: i=0, k=0 -> c=0.
        # C01: i=0, k=1 -> c=2.
        # C10: i=1, k=0 -> c=1.
        # C11: i=1, k=1 -> c=3.
        
        # So for block s (columns 2s, 2s+1):
        # C00 -> 2s
        # C01 -> 2s+2
        # C10 -> 2s+1
        # C11 -> 2s+3
        
        w_coeffs = [
            [1, 0, 0, 1],  # C00: m1, m4, m7
            [0, 1, 0, 0],  # C10: m2
            [0, 0, 1, 1],  # C01: m3, m5
            [1, -1, 1, 1]  # C11: m1, -m2, m3, m6
        ]
        # Wait, let's check C11 again.
        # C11 = m1 - m2 + m3 + m6.
        # Coeffs for m1..m7: [1, -1, 1, 0, 0, 1, 0]
        
        # Let's build the coefficient matrix W_block (4x7) where rows are C00, C10, C01, C11
        # and cols are m1..m7.
        W_block = np.zeros((4, 7), dtype=np.int32)
        # C00 = m1 + m4 - m5 + m7
        W_block[0, 0] = 1; W_block[0, 3] = 1; W_block[0, 4] = -1; W_block[0, 6] = 1
        # C10 = m2 + m4
        W_block[1, 1] = 1; W_block[1, 3] = 1
        # C01 = m3 + m5
        W_block[2, 2] = 1; W_block[2, 4] = 1
        # C11 = m1 - m2 + m3 + m6
        W_block[3, 0] = 1; W_block[3, 1] = -1; W_block[3, 2] = 1; W_block[3, 5] = 1
        
        for s in range(3):
            for i in range(7):
                col_idx = s * 7 + i
                # U
                U_sub[:, col_idx] = u_vecs[i]
                
                # V
                v_vec = np.zeros(12, dtype=np.int32)
                base = s * 4
                if i == 0: # B00+B11
                    v_vec[base+0] = 1; v_vec[base+3] = 1
                elif i == 1: # B00
                    v_vec[base+0] = 1
                elif i == 2: # B10-B11
                    v_vec[base+1] = 1; v_vec[base+3] = -1
                elif i == 3: # B01-B10
                    v_vec[base+2] = 1; v_vec[base+1] = -1
                elif i == 4: # B11
                    v_vec[base+3] = 1
                elif i == 5: # B00+B10
                    v_vec[base+0] = 1; v_vec[base+1] = 1
                elif i == 6: # B01+B11
                    v_vec[base+2] = 1; v_vec[base+3] = 1
                V_sub[:, col_idx] = v_vec
                
                # W
                w_vec = np.zeros(12, dtype=np.int32)
                w_base = s * 4
                w_vec[w_base+0] = W_block[0, i] # C00
                w_vec[w_base+1] = W_block[1, i] # C10
                w_vec[w_base+2] = W_block[2, i] # C01
                w_vec[w_base+3] = W_block[3, i] # C11
                W_sub[:, col_idx] = w_vec
        
        return U_sub, V_sub, W_sub, rank_sub

    # We have a 2x2 * 2x6 decomposition of rank 21.
    # Now we need to assemble the full 2x4 * 4x6 decomposition.
    # A (2x4) = [A_left (2x2), A_right (2x2)]
    # B (4x6) = [B_top (2x6); B_bottom (2x6)]
    # C = A_left * B_top + A_right * B_bottom
    
    # We can use the 2x2 * 2x6 decomposition twice:
    # 1. For A_left * B_top
    # 2. For A_right * B_bottom
    
    # Total rank = 21 + 21 = 42.
    # This is better than 48.
    
    U_left, V_top, W_1, rank_1 = build_2x2x2x6_decomp()
    U_right, V_bottom, W_2, rank_2 = build_2x2x2x6_decomp()
    
    # Now we need to map these local indices to global indices.
    # Global U (8x42)
    # A_left corresponds to A[0:2, 0:2] -> global U indices 0..3
    # A_right corresponds to A[0:2, 2:4] -> global U indices 4..7
    
    # Global V (24x42)
    # B_top corresponds to B[0:2, 0:6] -> global V indices 0..11
    # B_bottom corresponds to B[2:4, 0:6] -> global V indices 12..23
    
    # Global W (12x42)
    # C is 2x6. W indices are the same for both parts.
    
    U_full = np.zeros((U_dim, rank_1 + rank_2), dtype=np.int32)
    V_full = np.zeros((V_dim, rank_1 + rank_2), dtype=np.int32)
    W_full = np.zeros((W_dim, rank_1 + rank_2), dtype=np.int32)
    
    # Part 1: A_left * B_top
    # U_left is 4x21. Map to U_full[0:4, 0:21]
    U_full[0:4, 0:rank_1] = U_left
    # V_top is 12x21. Map to V_full[0:12, 0:21]
    V_full[0:12, 0:rank_1] = V_top
    # W_1 is 12x21. Map to W_full[0:12, 0:21]
    W_full[0:12, 0:rank_1] = W_1
    
    # Part 2: A_right * B_bottom
    # U_right is 4x21. Map to U_full[4:8, 21:42]
    U_full[4:8, rank_1:rank_1+rank_2] = U_right
    # V_bottom is 12x21. Map to V_full[12:24, 21:42]
    V_full[12:24, rank_1:rank_1+rank_2] = V_bottom
    # W_2 is 12x21. Map to W_full[0:12, 21:42]
    W_full[0:12, rank_1:rank_1+rank_2] = W_2
    
    # Verify the decomposition
    # T_reconstructed = sum_k U[:,k] V[:,k]^T W[:,k]^T ... no, it's a tensor.
    # T[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
    
    T_rec = np.zeros((U_dim, V_dim, W_dim), dtype=np.int32)
    for k in range(rank_1 + rank_2):
        T_rec += np.einsum('a,b,c->abc', U_full[:, k], V_full[:, k], W_full[:, k])
    
    # Check if T_rec == T
    if not np.array_equal(T_rec, T):
        # Fallback to schoolbook if verification fails
        best_rank = n * m * p
        U_school = np.zeros((U_dim, best_rank), dtype=np.int32)
        V_school = np.zeros((V_dim, best_rank), dtype=np.int32)
        W_school = np.zeros((W_dim, best_rank), dtype=np.int32)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
