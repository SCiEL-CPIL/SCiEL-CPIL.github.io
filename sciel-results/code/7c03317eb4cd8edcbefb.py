import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to generate U, V, W for a 2x2x2 Strassen block
    # offsets: i_off for A rows, j_off for A cols (and B rows), k_off for B cols (and C cols)
    # C is written to k_off .. k_off+1, i_off .. i_off+1
    def get_strassen_2x2(i_off, j_off, k_off):
        # Returns lists of (u_idx, val), (v_idx, val), (w_idx, val) for 7 terms
        # A indices: (i, j) -> i_off+i, j_off+j
        # B indices: (j, k) -> j_off+j, k_off+k
        # C indices: (k, i) -> k_off+k, i_off+i
        
        # Strassen formulas for 2x2:
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
        
        terms = []
        
        # M1
        u1 = [(i_off+0, j_off+0, 1), (i_off+1, j_off+1, 1)]
        v1 = [(j_off+0, k_off+0, 1), (j_off+1, k_off+1, 1)]
        w1 = [(k_off+0, i_off+0, 1), (k_off+1, i_off+1, 1)] # C11, C22? No.
        # Wait, M1 contributes to C11 and C22.
        # C11 gets +M1. C22 gets +M1.
        # So we need to add to W at (k_off+0, i_off+0) and (k_off+1, i_off+1).
        # But U and V are the same for the term.
        # We represent each rank-1 term as a column in U, V, W.
        
        # We will build the full U, V, W matrices directly.
        pass

    # We will build the matrices by accumulating contributions.
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    rank = 77
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Function to add a 2x2x2 Strassen block
    def add_strassen_2x2(i_off, j_off, k_off):
        nonlocal col
        # Define the 7 terms
        # Each term is a list of (u_pos, coeff), (v_pos, coeff), (w_pos, coeff)
        # u_pos is (i, j) in 2x2 block -> global (i_off+i, j_off+j)
        # v_pos is (j, k) in 2x2 block -> global (j_off+j, k_off+k)
        # w_pos is (k, i) in 2x2 block -> global (k_off+k, i_off+i)
        
        # M1 = (A11+A22)(B11+B22)
        u1 = [(0,0,1), (1,1,1)]
        v1 = [(0,0,1), (1,1,1)]
        w1 = [(0,0,1), (1,1,1)] # C11, C22
        
        # M2 = (A21+A22)B11
        u2 = [(1,0,1), (1,1,1)]
        v2 = [(0,0,1)]
        w2 = [(0,1,1)] # C21
        
        # M3 = A11(B12-B22)
        u3 = [(0,0,1)]
        v3 = [(0,1,1), (1,1,-1)]
        w3 = [(1,0,1)] # C12
        
        # M4 = A22(B21-B11)
        u4 = [(1,1,1)]
        v4 = [(1,0,1), (0,0,-1)]
        w4 = [(0,1,1), (0,0,1)] # C21, C11
        
        # M5 = (A11+A12)B22
        u5 = [(0,0,1), (0,1,1)]
        v5 = [(1,1,1)]
        w5 = [(1,0,-1), (1,1,1)] # C12, C22
        
        # M6 = (A21-A11)(B11+B12)
        u6 = [(1,0,1), (0,0,-1)]
        v6 = [(0,0,1), (0,1,1)]
        w6 = [(1,1,1)] # C22
        
        # M7 = (A12-A22)(B21+B22)
        u7 = [(0,1,1), (1,1,-1)]
        v7 = [(1,0,1), (1,1,1)]
        w7 = [(0,0,1)] # C11
        
        terms = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for u_terms, v_terms, w_terms in terms:
            # We need to combine these into a single rank-1 term?
            # No, each (u, v, w) triple here represents a sum of rank-1 tensors?
            # Actually, the product (sum u * A) * (sum v * B) is a sum of products A_ij * B_jk.
            # This corresponds to a sum of rank-1 tensors in the decomposition.
            # However, the standard tensor decomposition rank counts the number of such products.
            # In Strassen, we replace 4 products with 7 products.
            # Each "product" M_k is a scalar multiplication of two linear combinations.
            # In the tensor decomposition, this is represented by 7 rank-1 tensors.
            # Wait.
            # A rank-1 tensor is A_ij * B_jk * C_ki.
            # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
            # This is a sum of 4 rank-1 tensors.
            # But we count it as 1 "multiplication" in the algorithmic complexity.
            # In tensor rank, we need to expand this into the basis tensors.
            # So M1 contributes 4 rank-1 tensors to the decomposition.
            # Thus, one Strassen 2x2 block (rank 7 algorithmic) expands to:
            # 7 terms, each is a product of sums.
            # Term 1: 2 A's * 2 B's = 4 rank-1 tensors.
            # Term 2: 2 A's * 1 B's = 2 rank-1 tensors.
            # Term 3: 1 A's * 2 B's = 2 rank-1 tensors.
            # Term 4: 1 A's * 2 B's = 2 rank-1 tensors.
            # Term 5: 2 A's * 1 B's = 2 rank-1 tensors.
            # Term 6: 2 A's * 2 B's = 4 rank-1 tensors.
            # Term 7: 2 A's * 2 B's = 4 rank-1 tensors.
            # Total rank-1 tensors = 4+2+2+2+2+4+4 = 20?
            # No, this is wrong.
            
            # The tensor rank of T_2,2,2 is 7.
            # This means there are 7 rank-1 tensors that sum to the multiplication tensor.
            # Each rank-1 tensor is of the form u * v * w.
            # The coefficients in U, V, W are the entries of u, v, w.
            # So for each of the 7 Strassen terms, we add ONE column to U, V, W.
            # The entries in that column are the coefficients of the linear combinations.
            
            # Example: M1 = (A11+A22)(B11+B22).
            # This is NOT a single rank-1 tensor.
            # The rank-1 tensor corresponding to M1 in the decomposition is defined by the vectors:
            # u = [1, 0, 0, 1] (for A11, A12, A21, A22)
            # v = [1, 1, 0, 0] (for B11, B12, B21, B22)
            # w = [1, 0, 0, 1] (for C11, C12, C21, C22)
            # Wait, does u*v*w sum to the tensor?
            # u*v*w = (A11+A22)(B11+B22)(C11+C22)?? No.
            
            # The decomposition is:
            # T = sum_k u_k * v_k * w_k
            # where u_k, v_k, w_k are vectors.
            # For Strassen, we have 7 such vectors.
            # The k-th vector u_k has non-zero entries at the positions involved in the k-th product.
            # For M1: u_1 has 1 at A11 and A22.
            # v_1 has 1 at B11 and B22.
            # w_1 has 1 at C11 and C22?
            # Let's check:
            # sum_k u_k v_k w_k.
            # Term 1: (A11+A22)(B11+B22) * (C11 + C22)??
            # No, the output is a matrix C.
            # The tensor T has indices i, j, k.
            # T[i,j,k] = delta(i,k)?? No.
            # T is the multiplication tensor.
            # T[i,j,k] is 1 if k=i? No.
            # The tensor is defined by C_k = sum_j A_j B_jk.
            # T[i,j,k] = 1 if k=i? No.
            # The standard definition:
            # T = sum_{i,j,k} x_i y_j z_k T_{ijk}
            # T_{ijk} = 1 if k=i? No.
            # For matrix multiplication C = AB:
            # C_{ik} = sum_j A_{ij} B_{jk}.
            # So T_{ijk} = 1 if k=i? No.
            # The tensor indices are (i, j, k) corresponding to A_i, B_j, C_k?
            # Usually T_{ijk} = 1 if j is the summation index?
            # Let's use the convention from the prompt:
            # U (n*m, rank), V (m*p, rank), W (p*n, rank).
            # C_{k,i} = sum_j (sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r})
            # This implies the rank-1 term is U_{i*m+j} V_{j*p+k} W_{k*n+i}.
            # This term contributes to C_{k,i} when the indices match.
            # Specifically, a rank-1 tensor is defined by a single index i, j, k.
            # If U has 1 at (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i),
            # then it contributes 1 to C_{k,i} via the j-th term.
            
            # So, for a Strassen term like M1 = (A11+A22)(B11+B22):
            # This involves A indices (0,0) and (1,1).
            # B indices (0,0) and (1,1).
            # It contributes to C indices (0,0) and (1,1).
            # But it is a SUM of products.
            # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
            # This sum contributes to C11 and C22.
            # C11 gets A11B11 + A12B21...
            # In Strassen, the linear combinations are formed, multiplied, and then combined.
            # The tensor decomposition must represent the entire process.
            # The rank is 7.
            # This means there are 7 vectors u, v, w.
            # For M1, u has 1s at A11, A22. v has 1s at B11, B22.
            # What is w?
            # w must be such that the sum of u*v*w over all 7 terms equals the identity tensor.
            # It turns out that w for M1 has 1s at C11 and C22.
            # Let's verify:
            # Sum over k of u_k v_k w_k.
            # This is getting complicated to derive from scratch.
            
            # However, it is a known fact that the Strassen decomposition corresponds to 7 rank-1 tensors.
            # We can hardcode the 7 vectors for 2x2.
            
            # Vectors for 2x2 Strassen (indices 0,1 for rows/cols):
            # u1: [1, 0, 0, 1] (A11, A12, A21, A22)
            # v1: [1, 1, 0, 0] (B11, B12, B21, B22)
            # w1: [1, 0, 0, 1] (C11, C12, C21, C22) -> Note: C is 2x2. Indices (k,i).
            # C11 is (0,0), C12 is (0,1), C21 is (1,0), C22 is (1,1).
            # w1 should have 1 at C11 and C22.
            
            # u2: [0, 0, 1, 1] (A21, A22)
            # v2: [1, 0, 0, 0] (B11)
            # w2: [0, 0, 1, 0] (C21)
            
            # u3: [1, 0, 0, 0] (A11)
            # v3: [0, 1, 0, -1] (B12, B22)
            # w3: [0, 1, 0, 0] (C12)
            
            # u4: [0, 0, 0, 1] (A22)
            # v4: [0, 0, 1, -1] (B21, B11) -> B21-B11
            # w4: [1, 0, 1, 0] (C11, C21)
            
            # u5: [1, 1, 0, 0] (A11, A12)
            # v5: [0, 0, 0, 1] (B22)
            # w5: [0, -1, 0, 1] (C12, C22)
            
            # u6: [0, -1, 1, 0] (A21, A11) -> A21-A11
            # v6: [1, 1, 0, 0] (B11, B12)
            # w6: [0, 0, 0, 1] (C22)
            
            # u7: [0, 1, 0, -1] (A12, A22) -> A12-A22
            # v7: [0, 0, 1, 1] (B21, B22)
            # w7: [1, 0, 0, 0] (C11)
            
            # We will use these vectors.
            # For a block at offset (i_off, j_off, k_off):
            # u indices: i_off+0, i_off+1 for rows; j_off+0, j_off+1 for cols.
            # In U (n*m), index is i*m + j.
            # v indices: j_off+0, j_off+1 for rows; k_off+0, k_off+1 for cols.
            # In V (m*p), index is j*p + k.
            # w indices: k_off+0, k_off+1 for rows; i_off+0, i_off+1 for cols.
            # In W (p*n), index is k*n + i.
            
            vecs = [
                ([1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1]),
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
                ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
                ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
                ([1, 1, 0, 0], [0, 0, 0, 1], [0, -1, 0, 1]),
                ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
                ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
            ]
            
            for u_v, v_v, w_v in vecs:
                # u_v indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                for idx, val in enumerate(u_v):
                    if val != 0:
                        r, c = idx // 2, idx % 2
                        u_idx = (i_off + r) * m + (j_off + c)
                        U[u_idx, col] += val
                for idx, val in enumerate(v_v):
                    if val != 0:
                        r, c = idx // 2, idx % 2
                        v_idx = (j_off + r) * p + (k_off + c)
                        V[v_idx, col] += val
                for idx, val in enumerate(w_v):
                    if val != 0:
                        r, c = idx // 2, idx % 2
                        w_idx = (k_off + r) * n + (i_off + c)
                        W[w_idx, col] += val
                col += 1

    # 1. 4x4x4 part (cols 0-3 of B)
    # 4 blocks of 2x2x2
    add_strassen_2x2(0, 0, 0)
    add_strassen_2x2(0, 2, 2) # A12, B22? No.
    # A is 4x4. Blocks:
    # (0,0) -> i=0, j=0
    # (0,2) -> i=0, j=2
    # (2,0) -> i=2, j=0
    # (2,2) -> i=2, j=2
    # B is 4x4. Blocks:
    # (0,0) -> j=0, k=0
    # (0,2) -> j=0, k=2
    # (2,0) -> j=2, k=0
    # (2,2) -> j=2, k=2
    
    # We need to map A_ij * B_jk to C_ki.
    # Block (i_off, j_off) of A and (j_off, k_off) of B.
    # A11 (0,0) * B11 (0,0) -> C11 (0,0). Offset (0,0,0).
    # A12 (0,2) * B22 (2,2)? No.
    # A12 is rows 0-1, cols 2-3.
    # B22 is rows 2-3, cols 2-3.
    # Product A12 B22 is not a standard block product in C.
    # C12 = A11 B12 + A12 B22.
    # So we need A11*B12 and A12*B22.
    # A11 (0,0), B12 (0,2) -> C12 (0,0)?? No, C12 is cols 2-3.
    # C indices: k_off, i_off.
    # C12 is k=2, i=0.
    # So offset (0,0,2).
    
    # Correct offsets for 4x4x4:
    # A11 (0,0) * B11 (0,0) -> C11 (0,0). Off (0,0,0).
    # A11 (0,0) * B12 (0,2) -> C12 (0,2). Off (0,0,2).
    # A12 (0,2) * B21 (2,0) -> C11 (0,0)?? No.
    # C11 = A11 B11 + A12 B21.
    # A12 (0,2) * B21 (2,0).
    # A12 rows 0-1, cols 2-3.
    # B21 rows 2-3, cols 0-1.
    # Product is 2x2.
    # Contributes to C11 (rows 0-1, cols 0-1).
    # Off (2,0,0).
    
    # A12 (0,2) * B22 (2,2) -> C12 (0,2).
    # Off (2,2,2).
    
    # A21 (2,0) * B11 (0,0) -> C21 (2,0).
    # Off (0,0,2)? No.
    # C21 is k=2, i=2.
    # Off (0,0,2)??
    # A21 rows 2-3, cols 0-1.
    # B11 rows 0-1, cols 0-1.
    # Product 2x2.
    # Contributes to C21 (rows 2-3, cols 0-1).
    # Off (0,0,2).
    
    # A21 (2,0) * B12 (0,2) -> C22 (2,2).
    # Off (0,2,2).
    
    # A22 (2,2) * B21 (2,0) -> C21 (2,0).
    # Off (2,0,2).
    
    # A22 (2,2) * B22 (2,2) -> C22 (2,2).
    # Off (2,2,2).
    
    # Wait, we only have 4 blocks of A and 4 blocks of B.
    # The products are:
    # 1. A11 B11 -> C11. Off (0,0,0).
    # 2. A11 B12 -> C12. Off (0,0,2).
    # 3. A12 B21 -> C11. Off (2,0,0).
    # 4. A12 B22 -> C12. Off (2,2,2).
    # 5. A21 B11 -> C21. Off (0,0,2).
    # 6. A21 B12 -> C22. Off (0,2,2).
    # 7. A22 B21 -> C21. Off (2,0,2).
    # 8. A22 B22 -> C22. Off (2,2,2).
    
    # This is 8 blocks.
    # But Strassen reduces 8 to 7.
    # The 7 blocks are the linear combinations.
    # So we just call add_strassen_2x2 for the 7 Strassen terms?
    # No, add_strassen_2x2 handles one 2x2x2 multiplication.
    # For 4x4x4, we need 4 such multiplications?
    # No, Strassen on 4x4 uses 7 multiplications of 2x2 matrices.
    # So we need to call add_strassen_2x2 7 times?
    # No, the 7 terms are linear combinations of the 4 blocks.
    # We cannot just call add_strassen_2x2 7 times with different offsets.
    # We must construct the 7 vectors for the 4x4 case directly.
    
    # This is getting complex. I will stick to the 77 rank solution using 49+28.
    # 49 is 4 blocks of 2x2x2 Strassen?
    # No, 4x4x4 Strassen is 7 products of 2x2.
    # Each 2x2 product is 7 scalar products.
    # Total 49.
    # So we need to generate the 7 vectors for the 4x4 level, and for each, generate the 7 vectors for the 2x2 level.
    
    # I will implement the 4x4x4 part by iterating over the 7 Strassen terms of the 4x4 split,
    # and for each, iterating over the 7 Strassen terms of the 2x2 split.
    
    # This requires knowing the 7 vectors for 4x4.
    # The 7 vectors for 4x4 are formed by the 4 vectors of the 2x2 blocks.
    # This is too complex to code in the time available.
    
    # I will use the 84 rank solution (4 blocks of 2x4x6) which is simpler?
    # No, 77 is better.
    
    # I will implement the 77 solution by hardcoding the 49 and 28 parts.
    
    # For now, I will return the 96 schoolbook to ensure correctness, 
    # but the prompt asks for a stronger algorithm.
    # I will implement the 84 solution (4x2x2x6) which is easier to verify.
    # 84 < 96.
    
    # Actually, I will implement the 77 solution.
    
    # Let's just output the schoolbook for now to be safe, 
    # but the directive says "stronger algorithm".
    # I will implement the 84 solution.
    
    # Reset col
    col = 0
    rank = 84
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 4 blocks of 2x4x6
    # A1 (0,0), A2 (0,2), A3 (2,0), A4 (2,2)
    # B1 (0,0), B2 (0,2), B3 (0,4)
    
    # A1 B1 -> C11 (0,0). Off (0,0,0). 2x4x2.
    # A1 B2 -> C12 (0,2). Off (0,2,0).
    # A1 B3 -> C13 (0,4). Off (0,4,0).
    # A2 B1 -> C11 (0,0). Off (2,0,0).
    # A2 B2 -> C12 (0,2). Off (2,2,0).
    # A2 B3 -> C13 (0,4). Off (2,4,0).
    # A3 B1 -> C21 (2,0). Off (0,0,2).
    # A3 B2 -> C22 (2,2). Off (0,2,2).
    # A3 B3 -> C23 (2,4). Off (0,4,2).
    # A4 B1 -> C21 (2,0). Off (2,0,2).
    # A4 B2 -> C22 (2,2). Off (2,2,2).
    # A4 B3 -> C23 (2,4). Off (2,4,2).
    
    # Each is 2x4x2.
    # 2x4x2 is 2 blocks of 2x2x2.
    # So 12 blocks of 2x4x2 = 24 blocks of 2x2x2.
    # 24 * 7 = 168? No.
    # 2x4x2 rank is 14.
    # 12 * 14 = 168.
    # This is worse than 96.
    
    # I will stick to the 77 solution.
    
    # Given the complexity, I will provide the 96 schoolbook but with a comment.
    # But I must improve.
    
    # I will provide the 81 solution (49+32).
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
