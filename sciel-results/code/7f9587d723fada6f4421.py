"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 decomposition of 2x4 x 4x4 matrix multiplication using block-level Strassen.
    
    Partition: A is 2x4, B is 4x4, C = A*B is 2x4.
    Split B into four 2x2 blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3),
    B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3).
    
    The product can be viewed as multiplying two 2x2 'macro' matrices where each entry
    is a 2x2 block, using Strassen's 7-multiplication algorithm.
    
    Tensor convention: U has shape (n*m, rank) = (8, rank), V has shape (m*p, rank) = (16, rank),
    W has shape (p*n, rank) = (8, rank).
    
    Indexing:
    - U[i*m + j, c] for i in [0,n), j in [0,m)
    - V[j*p + k, c] for j in [0,m), k in [0,p)
    - W[k*n + i, c] for k in [0,p), i in [0,n)
    
    Here n=2, m=4, p=4.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # We'll build the decomposition using Strassen at the block level.
    # The 2x4 matrix A has entries A[i,j] for i in {0,1}, j in {0,1,2,3}.
    # The 4x4 matrix B has entries B[j,k] for j in {0,1,2,3}, k in {0,1,2,3}.
    # Product C[i,k] = sum_j A[i,j] * B[j,k].
    
    # Split columns of A and rows of B into two groups of 2:
    # Group 1: indices 0,1; Group 2: indices 2,3.
    
    # C[i,k] for k in {0,1} = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #                      = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # Group by which B block row:
    # = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    # This is like multiplying a 2x2 matrix [A[i,0], A[i,1]; A[i,2], A[i,3]] (for each i)
    # by a 4x2 matrix [B[0:2, k]; B[2:4, k]].
    
    # More naturally, think of it as: for each output row i and each pair of output columns (k1,k2),
    # we're doing a 2x2 * 2x2 multiplication if we group the middle index into pairs.
    
    # Let's use Strassen on the "block" structure. Consider the multiplication as:
    # A (2x4) = [A_left | A_right] where A_left is 2x2 (cols 0,1) and A_right is 2x2 (cols 2,3)
    # B (4x4) = [B_top_left, B_top_right; B_bottom_left, B_bottom_right]
    #           where each block is 2x2.
    
    # C = A * B = [A_left * B_top_left + A_right * B_bottom_left, A_left * B_top_right + A_right * B_bottom_right]
    
    # So C has two 2x2 blocks: C_left (cols 0,1) and C_right (cols 2,3).
    # C_left = A_left * B_top_left + A_right * B_bottom_left
    # C_right = A_left * B_top_right + A_right * B_bottom_right
    
    # Each of these is a sum of two 2x2 * 2x2 products. We can apply Strassen to each 2x2*2x2 product.
    # But we need to be careful: the Strassen decomposition applies to a single 2x2*2x2 product.
    
    # For a 2x2 * 2x2 product P = M * N, Strassen uses 7 scalar multiplications:
    # M = [[m00, m01], [m10, m11]], N = [[n00, n01], [n10, n11]]
    # M1 = (m00 + m11) * (n00 + n11)
    # M2 = (m10 + m11) * n00
    # M3 = m00 * (n01 - n11)
    # M4 = m11 * (n10 - n00)
    # M5 = (m00 + m01) * n11
    # M6 = (m10 - m00) * (n00 + n10)
    # M7 = (m01 - m11) * (n01 + n11)
    # P[0,0] = M1 + M4 - M5 + M7
    # P[0,1] = M3 + M5
    # P[1,0] = M2 + M4
    # P[1,1] = M1 - M2 + M3 + M6
    
    # For C_left = A_left * B_top_left + A_right * B_bottom_left, we need to combine two Strassen decompositions.
    # This gets complex. Let's instead directly enumerate the rank-28 terms.
    
    # Actually, let's think of it differently. The total rank is 28 = 4 * 7, suggesting we apply
    # Strassen to each of the four 2x2 * 2x2 block products independently, but the linearity means
    # we can't just do that naively because the Strassen intermediates for different blocks are distinct.
    
    # Let me define the 4 block multiplications:
    # Block 1: A_left (2x2) * B_top_left (2x2) -> contributes to C_left
    # Block 2: A_right (2x2) * B_bottom_left (2x2) -> contributes to C_left
    # Block 3: A_left (2x2) * B_top_right (2x2) -> contributes to C_right
    # Block 4: A_right (2x2) * B_bottom_right (2x2) -> contributes to C_right
    
    # For each 2x2*2x2 product, Strassen gives 7 terms. Total: 4*7 = 28 terms.
    
    # Let's index the columns of U, V, W from 0 to 27.
    # For block b (0 to 3), the 7 Strassen terms occupy columns b*7 to b*7+6.
    
    # Block 0: A_left * B_top_left
    #   A_left rows: i=0,1; cols j=0,1
    #   B_top_left rows: j=0,1; cols k=0,1
    # Block 1: A_right * B_bottom_left
    #   A_right rows: i=0,1; cols j=2,3
    #   B_bottom_left rows: j=2,3; cols k=0,1
    # Block 2: A_left * B_top_right
    #   A_left rows: i=0,1; cols j=0,1
    #   B_top_right rows: j=0,1; cols k=2,3
    # Block 3: A_right * B_bottom_right
    #   A_right rows: i=0,1; cols j=2,3
    #   B_bottom_right rows: j=2,3; cols k=2,3
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # For each block, define the mapping from (i, j) in A and (j, k) in B to the U, V, W indices.
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*4 + k
    # W index: k*n + i = k*2 + i
    
    # Strassen coefficients for a 2x2*2x2 product:
    # Term 0 (M1): (m00+m11)*(n00+n11) -> U: +1 for (0,0),(1,1); V: +1 for (0,0),(1,1); W: +1 for (0,0),(0,1),(1,0),(1,1) with signs
    # Let's define the Strassen structure more carefully.
    
    # For a 2x2 product P = M*N:
    # M = [[m00, m01], [m10, m11]], N = [[n00, n01], [n10, n11]]
    # 
    # M1 = (m00 + m11) * (n00 + n11)
    #   U part: m00 -> +1, m11 -> +1
    #   V part: n00 -> +1, n11 -> +1
    #   W part: P00 += M1, P01 += M1, P10 += M1, P11 += M1 ... wait, no.
    
    # Actually, the Strassen algorithm computes:
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # So for each Strassen term, we need to add appropriate coefficients to W.
    
    # Let me define the Strassen terms with their contributions to U, V, and W:
    # 
    # Term 0 (M1 = (m00+m11)*(n00+n11)):
    #   U: m00:+1, m11:+1
    #   V: n00:+1, n11:+1
    #   W: P00:+1, P11:+1
    #
    # Term 1 (M2 = (m10+m11)*n00):
    #   U: m10:+1, m11:+1
    #   V: n00:+1
    #   W: P10:+1, P11:-1
    #
    # Term 2 (M3 = m00*(n01-n11)):
    #   U: m00:+1
    #   V: n01:+1, n11:-1
    #   W: P01:+1, P11:+1
    #
    # Term 3 (M4 = m11*(n10-n00)):
    #   U: m11:+1
    #   V: n10:+1, n00:-1
    #   W: P00:+1, P10:+1
    #
    # Term 4 (M5 = (m00+m01)*n11):
    #   U: m00:+1, m01:+1
    #   V: n11:+1
    #   W: P00:-1, P01:+1
    #
    # Term 5 (M6 = (m10-m00)*(n00+n10)):
    #   U: m10:+1, m00:-1
    #   V: n00:+1, n10:+1
    #   W: P11:+1
    #
    # Term 6 (M7 = (m01-m11)*(n01+n11)):
    #   U: m01:+1, m11:-1
    #   V: n01:+1, n11:+1
    #   W: P00:+1
    
    # Now for each block, we map the local (m_ij, n_jk) to the global (i,j) and (j,k).
    
    # Block definitions:
    # Block 0: A_left * B_top_left
    #   m00=A[0,0], m01=A[0,1], m10=A[1,0], m11=A[1,1]
    #   n00=B[0,0], n01=B[0,1], n10=B[1,0], n11=B[1,1]
    #   Output: P00=C[0,0], P01=C[0,1], P10=C[1,0], P11=C[1,1]
    #
    # Block 1: A_right * B_bottom_left
    #   m00=A[0,2], m01=A[0,3], m10=A[1,2], m11=A[1,3]
    #   n00=B[2,0], n01=B[2,1], n10=B[3,0], n11=B[3,1]
    #   Output: P00=C[0,0], P01=C[0,1], P10=C[1,0], P11=C[1,1]
    #
    # Block 2: A_left * B_top_right
    #   m00=A[0,0], m01=A[0,1], m10=A[1,0], m11=A[1,1]
    #   n00=B[0,2], n01=B[0,3], n10=B[1,2], n11=B[1,3]
    #   Output: P00=C[0,2], P01=C[0,3], P10=C[1,2], P11=C[1,3]
    #
    # Block 3: A_right * B_bottom_right
    #   m00=A[0,2], m01=A[0,3], m10=A[1,2], m11=A[1,3]
    #   n00=B[2,2], n01=B[2,3], n10=B[3,2], n11=B[3,3]
    #   Output: P00=C[0,2], P01=C[0,3], P10=C[1,2], P11=C[1,3]
    
    # Define Strassen term structure: list of 7 tuples, each with:
    #   (U_offsets: [(local_i, local_j, coeff)], V_offsets: [(local_j, local_k, coeff)], W_offsets: [(local_out_i, local_out_k, coeff)])
    
    strassen_terms = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # For each block, define the mapping from local indices to global indices.
    # Local m indices: (li, lj) -> global A[i, j]
    # Local n indices: (lj, lk) -> global B[j, k]
    # Local output indices: (li, lk) -> global C[i, k]
    
    # Block 0: A_left * B_top_left
    # A[i, j] where i in {0,1}, j in {0,1}
    # B[j, k] where j in {0,1}, k in {0,1}
    # C[i, k] where i in {0,1}, k in {0,1}
    
    blocks = [
        # (j_offset_A, k_offset_B, i_offset_C, k_offset_C)
        # For block b, global j = j_local + j_offset, global k = k_local + k_offset
        # Global i = i_local + i_offset (but i_offset is always 0 here since n=2)
        # Global C k = k_local + k_offset_C
        {
            'j_offset': 0, 'k_offset': 0, 'i_offset': 0, 'k_offset_C': 0,
            # U index: i*4 + j = (i_local + 0)*4 + (j_local + 0)
            # V index: j*4 + k = (j_local + 0)*4 + (k_local + 0)
            # W index: k*2 + i = (k_local + 0)*2 + (i_local + 0)
        },
        {
            'j_offset': 2, 'k_offset': 0, 'i_offset': 0, 'k_offset_C': 0,
            # A[i, j] with j in {2,3}, B[j, k] with j in {2,3}, k in {0,1}
            # C[i, k] with i in {0,1}, k in {0,1}
        },
        {
            'j_offset': 0, 'k_offset': 2, 'i_offset': 0, 'k_offset_C': 2,
            # A[i, j] with j in {0,1}, B[j, k] with j in {0,1}, k in {2,3}
            # C[i, k] with i in {0,1}, k in {2,3}
        },
        {
            'j_offset': 2, 'k_offset': 2, 'i_offset': 0, 'k_offset_C': 2,
            # A[i, j] with j in {2,3}, B[j, k] with j in {2,3}, k in {2,3}
            # C[i, k] with i in {0,1}, k in {2,3}
        },
    ]
    
    for b_idx, block in enumerate(blocks):
        j_off = block['j_offset']
        k_off = block['k_offset']
        i_off = block['i_offset']
        k_off_C = block['k_offset_C']
        
        for t_idx, (u_offsets, v_offsets, w_offsets) in enumerate(strassen_terms):
            col = b_idx * 7 + t_idx
            
            for (li, lj, coeff) in u_offsets:
                gi = li + i_off
                gj = lj + j_off
                u_idx = gi * m + gj  # i*4 + j
                U[u_idx][col] += coeff
            
            for (lj, lk, coeff) in v_offsets:
                gj = lj + j_off
                gk = lk + k_off
                v_idx = gj * p + gk  # j*4 + k
                V[v_idx][col] += coeff
            
            for (li, lk, coeff) in w_offsets:
                gi = li + i_off
                gk = lk + k_off_C
                w_idx = gk * n + gi  # k*2 + i
                W[w_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
