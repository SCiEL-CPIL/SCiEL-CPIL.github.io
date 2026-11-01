"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    def time_left():
        return deadline - time.time()
    
    # Build the full multiplication tensor T[a, b, c]
    # T[a, b, c] = 1 if A index a and B index b contribute to C index c
    # a = i*m + j where i in [0,n), j in [0,m)
    # b = j*p + k where j in [0,m), k in [0,p)
    # c = k*n + i
    # So T[a, b, c] = 1 when the j in a equals the j in b, and c = k*n + i
    
    # Generate all (a, b, c) triples
    tensor_triples = []
    for a in range(A_dim):
        i = a // m
        j = a % m
        for b in range(B_dim):
            j2 = b // p
            k = b % p
            if j == j2:
                c = k * n + i
                tensor_triples.append((a, b, c))
    
    # Schoolbook: rank 64
    rank_schoolbook = len(tensor_triples)  # 64
    
    # Strassen-based decomposition: rank 56
    # Strategy: Split A into left (cols 0,1) and right (cols 2,3)
    # Split B into 4 blocks of 2 columns each (cols 0-1, 2-3, 4-5, 6-7)
    # Each (2x2) * (2x2) = 2x2 multiplication uses Strassen (rank 7)
    # For each of the 4 B-blocks, we need:
    #   A_left (2x2) * B_block_top (2x2) -> contributes to C rows
    #   A_right (2x2) * B_block_bottom (2x2) -> contributes to C rows
    # Wait, let me reconsider the block structure.
    #
    # A is 2x4: rows i=0,1; cols j=0,1,2,3
    # B is 4x8: rows j=0,1,2,3; cols k=0,...,7
    # C is 8x2: rows k=0,...,7; cols i=0,1
    # C[k,i] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # Split A into A_L (cols 0,1) and A_R (cols 2,3), each 2x2
    # Split B into 4 blocks B_g (cols 2g, 2g+1), each 4x2
    # C_g (rows 2g, 2g+1) = A_L * B_g_top + A_R * B_g_bottom
    # where B_g_top = B[0:2, 2g:2g+2] (2x2)
    #       B_g_bottom = B[2:4, 2g:2g+2] (2x2)
    #
    # So for each g in {0,1,2,3}:
    #   C_{2g, 0} = A[0,0]*B[0,2g] + A[0,1]*B[1,2g] + A[0,2]*B[2,2g] + A[0,3]*B[3,2g]
    #   C_{2g, 1} = A[1,0]*B[0,2g] + A[1,1]*B[1,2g] + A[1,2]*B[2,2g] + A[1,3]*B[3,2g]
    #   C_{2g+1, 0} = A[0,0]*B[0,2g+1] + A[0,1]*B[1,2g+1] + A[0,2]*B[2,2g+1] + A[0,3]*B[3,2g+1]
    #   C_{2g+1, 1} = A[1,0]*B[0,2g+1] + A[1,1]*B[1,2g+1] + A[1,2]*B[2,2g+1] + A[1,3]*B[3,2g+1]
    #
    # This is: C_block_g = A_L * B_g_top + A_R * B_g_bottom
    # where C_block_g is 2x2, A_L is 2x2, B_g_top is 2x2, A_R is 2x2, B_g_bottom is 2x2
    #
    # Each 2x2 * 2x2 = 2x2 can be done with rank 7 (Strassen)
    # So total rank = 4 * (7 + 7) = 56
    
    # Strassen for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each Strassen term is a rank-1 contribution
    # to the 2x2*2x2=2x2 tensor.
    # We need to map these to the full tensor.
    
    # Let's build the rank-56 decomposition.
    # For each g in {0,1,2,3}, we have two Strassen multiplications:
    #   1. A_L * B_g_top (both 2x2)
    #   2. A_R * B_g_bottom (both 2x2)
    
    # Strassen terms for a 2x2*2x2:
    # Each term is of the form (linear combo of A entries) * (linear combo of B entries) -> contributes to C entries
    # In tensor form, each term gives a rank-1 tensor: u * v * w
    # where u is in R^4 (A side), v in R^4 (B side), w in R^4 (C side)
    
    # For a 2x2 matrix A with entries A00, A01, A10, A11
    # and B with entries B00, B01, B10, B11
    # C = A*B has entries C00, C01, C10, C11
    
    # Strassen rank-1 terms (u, v, w) where:
    # u corresponds to the linear combo of A entries
    # v corresponds to the linear combo of B entries  
    # w corresponds to the linear combo of C entries (with signs)
    
    # M1 = (A00+A11)*(B00+B11): 
    #   u = (1,0,0,1) [A00+A11]
    #   v = (1,0,0,1) [B00+B11]
    #   w = (1,0,0,-1) [contributes +M1 to C00, -M1 to C11]
    #   Actually C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6
    #   So M1 contributes +1 to C00 and +1 to C11.
    #   w = (1,0,0,1)
    
    # M2 = (A10+A11)*B00:
    #   u = (0,0,1,1) [A10+A11]
    #   v = (1,0,0,0) [B00]
    #   w = (0,0,1,-1) [contributes to C10 and -C11]
    
    # M3 = A00*(B10-B11):
    #   u = (1,0,0,0) [A00]
    #   v = (0,0,1,-1) [B10-B11]
    #   w = (0,1,0,1) [contributes to C01 and C11]
    
    # M4 = A11*(B01-B10):
    #   u = (0,0,0,1) [A11]
    #   v = (0,1,-1,0) [B01-B10]
    #   w = (1,0,1,0) [contributes to C00 and C10]
    
    # M5 = (A00+A01)*B11:
    #   u = (1,1,0,0) [A00+A01]
    #   v = (0,0,0,1) [B11]
    #   w = (-1,1,0,0) [contributes -M5 to C00, +M5 to C01]
    
    # M6 = (A10-A00)*(B00+B10):
    #   u = (-1,0,1,0) [A10-A00]
    #   v = (1,0,1,0) [B00+B10]
    #   w = (0,0,0,1) [contributes to C11]
    
    # M7 = (A01-A11)*(B01+B11):
    #   u = (0,1,0,-1) [A01-A11]
    #   v = (0,1,0,1) [B01+B11]
    #   w = (1,0,0,0) [contributes to C00]
    
    # Verify: C00 = M1+M4-M5+M7 -> w contributions: 1+1-1+1 = 2? No, each w is a coefficient vector.
    # The total C00 coefficient from each term:
    # M1: w[0]=1 -> +1
    # M4: w[0]=1 -> +1
    # M5: w[0]=-1 -> -1
    # M7: w[0]=1 -> +1
    # Total C00 = 1+1-1+1 = 2? That's wrong.
    # 
    # I think the w vector should represent the output coefficients directly.
    # Let me redefine: each rank-1 term contributes u_i * v_j to sum_k w_k * C_k
    # So C = sum_terms u * v * w^T
    # For M1: u=(1,0,0,1), v=(1,0,0,1), and it contributes to C00 and C11.
    # C00 gets M1, C11 gets M1. So w=(1,0,0,1).
    # For M4: contributes to C00 and C10. w=(1,0,1,0).
    # For M5: contributes -M5 to C00, +M5 to C01. w=(-1,1,0,0).
    # For M7: contributes to C00. w=(1,0,0,0).
    # C00 total: 1 (from M1) + 1 (from M4) + (-1) (from M5) + 1 (from M7) = 2. Still wrong.
    #
    # The issue: Strassen gives C = sum of 7 products, and each product appears in the final sum with coefficient +1.
    # So the tensor is T = sum_s u_s * v_s * w_s^T
    # where w_s is the vector of coefficients for that term in the output.
    #
    # C00 = M1 + M4 - M5 + M7
    # So the coefficient of M1 in C00 is +1, M4 is +1, M5 is -1, M7 is +1.
    # This means:
    # Term M1: w=(1, 0, 0, 1) because M1 appears in C00 (coeff +1) and C11 (coeff +1)
    # Term M4: w=(1, 0, 1, 0) because M4 appears in C00 (coeff +1) and C10 (coeff +1)
    # Term M5: w=(-1, 1, 0, 0) because M5 appears in C00 (coeff -1) and C01 (coeff +1)
    # Term M7: w=(1, 0, 0, 0) because M7 appears only in C00 (coeff +1)
    #
    # Let me verify C00: from M1: +1, M4: +1, M5: -1, M7: +1. Sum = 2. But should be 1.
    #
    # I see the error. In Strassen:
    # C00 = M1 - M5 + M7 + M4  (this is what I had)
    # But M1 = (A00+A11)(B00+B11)
    # M5 = (A00+A01)(B11)
    # M7 = (A01-A11)(B01+B11)
    # M4 = A11(B01-B10)
    #
    # Let's expand:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # -M5 = -A00B11 - A01B11
    # M7 = A01B01 + A01B11 - A11B01 - A11B11
    # M4 = A11B01 - A11B10
    #
    # C00 should be A00B00 + A01B10
    #
    # Sum: A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11 + A11B01 - A11B10
    # = A00B00 + A11B00 + A01B01 - A01B11 + A11B11 - A11B11 + A01B11 - A11B10
    # Hmm, this is getting messy. Let me just use the standard Strassen formula and trust it.
    #
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So the w vectors are:
    # M1: (1, 0, 0, 1)
    # M2: (0, 0, 1, -1)
    # M3: (0, 1, 0, 1)
    # M4: (1, 0, 1, 0)
    # M5: (-1, 1, 0, 0)
    # M6: (0, 0, 0, 1)
    # M7: (1, 0, 0, 0)
    
    # And u, v vectors:
    # M1: u=(1,0,0,1), v=(1,0,0,1)
    # M2: u=(0,0,1,1), v=(1,0,0,0)
    # M3: u=(1,0,0,0), v=(0,0,1,-1)
    # M4: u=(0,0,0,1), v=(0,1,-1,0)
    # M5: u=(1,1,0,0), v=(0,0,0,1)
    # M6: u=(-1,0,1,0), v=(1,0,1,0)
    # M7: u=(0,1,0,-1), v=(0,1,0,1)
    
    strassen_terms = [
        # (u, v, w)
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),    # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),    # M2
        ((1,0,0,0), (0,0,1,-1), (0,1,0,1)),    # M3
        ((0,0,0,1), (0,1,-1,0), (1,0,1,0)),    # M4
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),    # M5
        ((-1,0,1,0), (1,0,1,0), (0,0,0,1)),    # M6
        ((0,1,0,-1), (0,1,0,1), (1,0,0,0)),    # M7
    ]
    
    # Now build the full rank-56 decomposition
    # For each g in {0,1,2,3}:
    #   A_L = A[:, 0:2] (cols 0,1): indices in A: [0,1,4,5] corresponding to (i,j) = (0,0),(0,1),(1,0),(1,1)
    #   A_R = A[:, 2:4] (cols 2,3): indices in A: [2,3,6,7]
    #   B_g_top = B[0:2, 2g:2g+2]: indices in B: [2g, 2g+1, 8+2g, 9+2g]
    #   B_g_bottom = B[2:4, 2g:2g+2]: indices in B: [16+2g, 17+2g, 24+2g, 25+2g]
    #   C_block = C[2g:2g+2, :] (rows 2g, 2g+1): indices in C: [4g, 4g+1, 4g+2, 4g+3]
    #     C[4g, 0] = C[2g, 0] -> flat index 2g*2+0 = 4g
    #     C[4g+1, 0] = C[2g+1, 0] -> flat index (2g+1)*2+0 = 4g+2
    #     C[4g, 1] = C[2g, 1] -> flat index 2g*2+1 = 4g+1
    #     C[4g+1, 1] = C[2g+1, 1] -> flat index (2g+1)*2+1 = 4g+3
    #   So C block flat indices: [4g, 4g+1, 4g+2, 4g+3] corresponding to (C00, C01, C10, C11) of the block
    
    # For the 2x2 sub-tensor, the local indices are:
    # A_L local: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 -> flat A indices [0, 1, 4, 5]
    # A_R local: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 -> flat A indices [2, 3, 6, 7]
    # B_g_top local: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 -> flat B indices [2g, 2g+1, 8+2g, 9+2g]
    # B_g_bottom local: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 -> flat B indices [16+2g, 17+2g, 24+2g, 25+2g]
    # C_block local: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 -> flat C indices [4g, 4g+1, 4g+2, 4g+3]
    
    U_list = []
    V_list = []
    W_list = []
    
    for g in range(4):
        # A_L indices
        A_L_idx = [0, 1, 4, 5]
        # A_R indices
        A_R_idx = [2, 3, 6, 7]
        # B_g_top indices
        B_top_idx = [2*g, 2*g+1, 8+2*g, 9+2*g]
        # B_g_bottom indices
        B_bot_idx = [16+2*g, 17+2*g, 24+2*g, 25+2*g]
        # C_block indices
        C_idx = [4*g, 4*g+1, 4*g+2, 4*g+3]
        
        for (u, v, w) in strassen_terms:
            # A_L * B_g_top
            u_full = [0.0] * A_dim
            for li, flat in enumerate(A_L_idx):
                u_full[flat] = u[li]
            v_full = [0.0] * B_dim
            for li, flat in enumerate(B_top_idx):
                v_full[flat] = v[li]
            w_full = [0.0] * C_dim
            for li, flat in enumerate(C_idx):
                w_full[flat] = w[li]
            U_list.append(u_full)
            V_list.append(v_full)
            W_list.append(w_full)
            
            # A_R * B_g_bottom
            u_full2 = [0.0] * A_dim
            for li, flat in enumerate(A_R_idx):
                u_full2[flat] = u[li]
            v_full2 = [0.0] * B_dim
            for li, flat in enumerate(B_bot_idx):
                v_full2[flat] = v[li]
            w_full2 = [0.0] * C_dim
            for li, flat in enumerate(C_idx):
                w_full2[flat] = w[li]
            U_list.append(u_full2)
            V_list.append(v_full2)
            W_list.append(w_full2)
    
    rank_strassen = len(U_list)  # 56
    
    # Verify the decomposition
    U_mat = np.array(U_list).T  # shape (A_dim, rank)
    V_mat = np.array(V_list).T  # shape (B_dim, rank)
    W_mat = np.array(W_list).T  # shape (C_dim, rank)
    
    # Check: T[a,b,c] should equal sum_s U[a,s]*V[b,s]*W[c,s]
    # Build T
    T = np.zeros((A_dim, B_dim, C_dim))
    for (a, b, c) in tensor_triples:
        T[a, b, c] = 1.0
    
    # Compute reconstructed tensor
    T_rec = np.einsum('as,bs,cs->abc', U_mat, V_mat, W_mat)
    
    if not np.allclose(T, T_rec, atol=1e-10):
        # Fall back to schoolbook
        U_school = np.zeros((A_dim, rank_schoolbook))
        V_school = np.zeros((B_dim, rank_schoolbook))
        W_school = np.zeros((C_dim, rank_schoolbook))
        for col, (a, b, c) in enumerate(tensor_triples):
            U_school[a, col] = 1
            V_school[b, col] = 1
            W_school[c, col] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
