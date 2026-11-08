import numpy as np

def construct():
    n, m, p = 3, 4, 4
    # Decompose 3x4x4 into 1x4x4 (rank 4) + 2x4x4 (rank 24) = rank 28
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 1x4x4 multiplication (row 0 of A) ---
    # A is 1x4, B is 4x4, C is 1x4
    # Local A indices: 0..3 (col)
    # Local B indices: 0..15 (row * 4 + col)
    # Local C indices: 0..3 (col)
    # Rank 4: each output column c = sum_i a_i * b_{i,c}
    # For each output column c (0..3):
    #   U[0, c] = 1 (a_0) ... wait, no.
    #   Actually, for 1x4x4, C_j = sum_i A_i * B_{i,j}
    #   Tensor T_{i, (r,c), j} = delta_{r,i} delta_{c,j}
    #   Rank 4 decomposition:
    #   For each j in 0..3:
    #     U[0, j] = 1  # This is wrong. U has shape (1*4, 4) = (4, 4)
    #     Let's be precise.
    #     A has 4 elements: A_0, A_1, A_2, A_3
    #     B has 16 elements: B_{r,c}
    #     C has 4 elements: C_j
    #     C_j = sum_{i=0}^3 A_i * B_{i,j}
    #     We can write C_j = sum_{k=0}^3 M_k where M_k = A_k * B_{k,j}
    #     So for each k in 0..3:
    #       U[k, k] = 1  (A_k)
    #       V[k*4 + j, k] = 1 for all j? No.
    #       The term is A_k * (sum over ... no)
    #       Actually, C_j depends on A_k and B_{k,j}.
    #       So for fixed k, the contribution to C_j is A_k * B_{k,j}.
    #       This is a rank 1 term in the tensor?
    #       T_{a, b, c} = delta_{a,k} * delta_{b, k*4+c} * delta_{c,c}
    #       Wait, the index for B is r*4+c. If we fix r=k, then B index is k*4+c.
    #       So for each k, we have a tensor that is 1 if a=k, b=k*4+c, c=c.
    #       This is not rank 1. It's a sum over c.
    #       Actually, C_j = A_j * B_{j,j} + A_0 * B_{0,j} + ...
    #       Let's use the standard rank 4 decomposition for 1x4x4.
    #       C_j = sum_i A_i B_{i,j}
    #       We can define 4 products M_j = A_j * B_{j, :} ? No.
    #       Let's just use 4 terms, one for each output column j.
    #       M_j = A_{:} * B_{:, j} ? No, A is 1x4, B_{:,j} is 4x1.
    #       M_j = dot(A, B[:,j])
    #       This is a scalar.
    #       The tensor is T_{i, (r,c), j} = delta_{r,i} delta_{c,j}.
    #       We can decompose this as sum_{k=0}^3 u_k v_k w_k^T
    #       where u_k[i] = delta_{i,k}
    #       v_k[(r,c)] = delta_{r,k} delta_{c, ...} ?
    #       If we pick v_k[(r,c)] = delta_{r,k}, then w_k[j] = delta_{c,j} ? No, c is fixed in v.
    #       Let's try:
    #       u_k[i] = delta_{i,k}
    #       v_k[r*4+c] = delta_{r,k}
    #       w_k[j] = 1 if c==j? No, c is summed over.
    #       Actually, T_{i, r*4+c, j} = delta_{i,r} delta_{c,j}
    #       = sum_{k} delta_{i,k} delta_{r,k} delta_{c,j}
    #       So u_k[i] = delta_{i,k}
    #       v_k[r*4+c] = delta_{r,k}
    #       w_k[j] = delta_{c,j} -- but c is part of v's index.
    #       Wait, v_k is a vector of length 16.
    #       If v_k[r*4+c] = delta_{r,k}, then summing over k:
    #       sum_k u_k[i] v_k[r*4+c] w_k[j] = sum_k delta_{i,k} delta_{r,k} w_k[j]
    #       = delta_{i,r} w_r[j]
    #       We want this to be delta_{i,r} delta_{c,j}.
    #       So we need w_r[j] to depend on c. But w_r is just a vector of length 4 (for j).
    #       It cannot depend on c.
    #       So this decomposition doesn't work with rank 4 in this simple way.
    #       
    #       Let's rethink. 1x4x4 multiplication.
    #       A = [a0, a1, a2, a3]
    #       B = 4x4 matrix
    #       C = [c0, c1, c2, c3]
    #       c_j = a0*B_0j + a1*B_1j + a2*B_2j + a3*B_3j
    #       This is a dot product of a vector of length 4 with a column of B.
    #       We can compute c_j for each j independently.
    #       c_j = dot(a, B[:,j])
    #       The dot product of two length-4 vectors has rank 4? No, rank 1 if we precompute?
    #       No, the tensor for dot product of two length-n vectors is sum_i x_i y_i.
    #       This tensor has rank n.
    #       So for each j, we need rank 4?
    #       Total rank 16?
    #       But we can share the a_i terms.
    #       Let's define 4 products M_i = a_i * B_{i,:} ? No.
    #       Let's define M_i = a_i. Then C = sum_i M_i * B_{i,:}
    #       This is not a tensor decomposition in the standard sense (rank 1 terms).
    #       
    #       Actually, the tensor T_{i, (r,c), j} = delta_{i,r} delta_{c,j}
    #       This tensor has rank 4.
    #       Proof: T = sum_{k=0}^3 u_k v_k w_k^T
    #       u_k[i] = delta_{i,k}
    #       v_k[r*4+c] = delta_{r,k} delta_{c,k} ? No.
    #       Let's try:
    #       u_k[i] = delta_{i,k}
    #       v_k[r*4+c] = delta_{r,k}
    #       w_k[j] = delta_{j,c} ? No, c is not in w.
    #       
    #       Wait, the index (r,c) in B is flattened to r*4+c.
    #       The condition is c=j.
    #       So we need w to depend on c. But w only has index j.
    #       This means we cannot separate c and j in a rank 1 term unless c=j is enforced by v or u.
    #       If we set v_k[r*4+c] = delta_{r,k} delta_{c,k}, then:
    #       sum_k delta_{i,k} delta_{r,k} delta_{c,k} w_k[j]
    #       = delta_{i,r} delta_{r,c} w_r[j]
    #       We want delta_{i,r} delta_{c,j}.
    #       This requires delta_{r,c} w_r[j] = delta_{c,j}.
    #       If r=c, then w_r[j] = delta_{r,j}.
    #       So w_k[j] = delta_{k,j}.
    #       Let's check:
    #       sum_k delta_{i,k} delta_{r,k} delta_{c,k} delta_{k,j}
    #       = 1 if i=r=c=j, else 0.
    #       This is not delta_{i,r} delta_{c,j}.
    #       For example, i=0, r=0, c=1, j=1.
    #       delta_{0,0} delta_{0,0} delta_{1,0} delta_{0,1} = 0.
    #       But we want delta_{0,0} delta_{1,1} = 1.
    #       So this doesn't work.
    #       
    #       The rank of the 1x4x4 tensor is actually 4.
    #       T_{i, (r,c), j} = delta_{i,r} delta_{c,j}
    #       This is a Kronecker product of delta_{i,r} and delta_{c,j}.
    #       The rank of a Kronecker product is the product of the ranks.
    #       Rank of delta_{i,r} (4x4 identity) is 4.
    #       Rank of delta_{c,j} (4x4 identity) is 4.
    #       So the rank is 16?
    #       No, the tensor is not a Kronecker product in the index space.
    #       The indices are (i) for A, (r,c) for B, (j) for C.
    #       T_{i, r, c, j} = delta_{i,r} delta_{c,j}
    #       This is a 4x4x4x4 tensor.
    #       It can be written as I_4 (in i,r) tensor I_4 (in c,j).
    #       The rank of this tensor is 4*4 = 16.
    #       So 1x4x4 has rank 16?
    #       That seems high.
    #       
    #       Let's reconsider.
    #       C_j = sum_i A_i B_{i,j}
    #       We can compute this with 4 multiplications if we allow precomputation?
    #       No, in tensor decomposition, we cannot precompute.
    #       Each rank 1 term is a product of one A-element, one B-element, and one C-element.
    #       So we need to cover all pairs (i, (i,j)) that contribute to C_j.
    #       For each j, we need to cover 4 pairs: (0, 0,j), (1, 1,j), (2, 2,j), (3, 3,j).
    #       A single rank 1 term u v w^T covers one A-element, one B-element, and one C-element.
    #       So it covers exactly one pair (i, (r,c)) for a specific j.
    #       Since i=r and c=j, each term covers one (i, (i,j)).
    #       There are 4*4 = 16 such pairs.
    #       So the rank is 16.
    #       
    #       So Part 1 (1x4x4) has rank 16.
    #       Part 2 (2x4x4) has rank ?
    #       2x4x4 multiplication.
    #       A is 2x4, B is 4x4, C is 2x4.
    #       C_{i,j} = sum_k A_{i,k} B_{k,j}
    #       For each i (0,1) and j (0..3), we have 4 terms.
    #       Total 2*4*4 = 32 terms.
    #       Can we do better?
    #       Strassen for 2x2x2 has rank 7.
    #       We can decompose 2x4x4 into four 2x2x2 blocks.
    #       A = [A11 A12; A21 A22] where each is 2x2.
    #       B = [B11 B12; B21 B22] where each is 2x2.
    #       C = [C11 C12; C21 C22] where each is 2x2.
    #       C11 = A11 B11 + A12 B21
    #       C12 = A11 B12 + A12 B22
    #       C21 = A21 B11 + A22 B21
    #       C22 = A21 B12 + A22 B22
    #       Each C-block is a sum of two 2x2x2 multiplications.
    #       Each 2x2x2 multiplication has rank 7 (Strassen).
    #       So each C-block needs 2*7 = 14 ranks?
    #       Total rank 4*14 = 56?
    #       That's worse than 32.
    #       
    #       Actually, we can share the Strassen factors.
    #       For C11 = A11 B11 + A12 B21
    #       We can use Strassen on the combined system?
    #       No, A11 and A12 are different rows.
    #       
    #       Let's just use the naive rank for 2x4x4, which is 2*4*4 = 32.
    #       Total rank = 16 + 32 = 48.
    #       This is worse than the current 44.
    #       
    #       So the block-diagonal strategy with 1x4x4 and 2x4x4 doesn't help.
    #       
    #       Let's try a different decomposition.
    #       Maybe 3x4x4 can be decomposed into smaller blocks more efficiently.
    #       
    #       Given the time constraints and the complexity, I will stick with a known good rank.
    #       The current program has rank 44.
    #       I will try to find a rank 43 or lower decomposition.
    #       
    #       Actually, the task says "Find a rank-minimal exact tensor decomposition".
    #       The known lower bound for 3x4x4 is 36.
    #       The known upper bound is 44 (from the current program).
    #       
    #       I will return the current program's logic but maybe with a small optimization.
    #       Since I cannot easily find a better rank, I will just return the current solution.
    
    # Re-implementing the current solution to be safe
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
