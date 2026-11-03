import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # We will build the U, V, W matrices for the whole problem
    # by concatenating the factors from sub-problems.
    
    # Sub-problem decomposition:
    # Split n=3 into n1=1, n2=2
    # Split m=4 into m1=2, m2=2
    # Split p=4 into p1=2, p2=2
    
    # Sub-problems:
    # 1. A11 (1x2) * B11 (2x2) -> C11 (1x2)  [Type 1,2,2]
    # 2. A11 (1x2) * B12 (2x2) -> C12 (1x2)  [Type 1,2,2]
    # 3. A12 (1x2) * B21 (2x2) -> C13 (1x2)  [Type 1,2,2]
    # 4. A12 (1x2) * B22 (2x2) -> C14 (1x2)  [Type 1,2,2]
    # 5. A21 (2x2) * B11 (2x2) -> C21 (2x2)  [Type 2,2,2]
    # 6. A21 (2x2) * B12 (2x2) -> C22 (2x2)  [Type 2,2,2]
    # 7. A22 (2x2) * B21 (2x2) -> C23 (2x2)  [Type 2,2,2]
    # 8. A22 (2x2) * B22 (2x2) -> C24 (2x2)  [Type 2,2,2]
    
    # Note: The indices for C in the sub-problems need to be mapped back to the global C.
    # C is 3x4.
    # Row 0 of C comes from sub-problems 1 and 2.
    #   Sub-problem 1 gives C11 (cols 0,1 of C row 0)
    #   Sub-problem 2 gives C12 (cols 2,3 of C row 0)
    # Row 1,2 of C come from sub-problems 5,6,7,8.
    #   Sub-problem 5 gives C21 (rows 0,1 of C sub-block, cols 0,1 of C sub-block)
    #   Wait, the mapping is:
    #   A21 is rows 1,2 of A. B11 is cols 0,1 of B (since m is split, B is m x p, so B11 is rows 0,1 of B).
    #   Result C21 is 2x2. It corresponds to C rows 1,2 and cols 0,1.
    
    # Let's define the sub-problems and their global indices.
    
    sub_problems = []
    
    # Helper to add a sub-problem
    # type: 0 for 2x2x2, 1 for 1x2x2
    # a_rows: global row indices of A
    # b_cols: global col indices of B (which are rows of B in the tensor sense? No, B is m x p. m is inner. So B indices are (j, k). j is 0..3, k is 0..3.)
    # c_rows: global row indices of C
    # c_cols: global col indices of C
    
    # Split A rows: A1: [0], A2: [1, 2]
    # Split B cols (inner dim m): B1: [0, 1], B2: [2, 3]
    # Split C cols (outer dim p): C1: [0, 1], C2: [2, 3]
    
    # Sub-problem 1: A1 (1x2) * B1 (2x2)
    # A rows: [0]
    # B inner: [0, 1]
    # C rows: [0]
    # C cols: [0, 1]
    sub_problems.append({
        'type': 1,
        'a_rows': [0],
        'b_inner': [0, 1],
        'c_rows': [0],
        'c_cols': [0, 1]
    })
    
    # Sub-problem 2: A1 (1x2) * B1 (2x2) -> but B1 is first half of m?
    # Wait, my split of m was [0,1] and [2,3].
    # My split of p was [0,1] and [2,3].
    #
    # A1 is 1x4. Split into A11 (1x2) and A12 (1x2).
    # B is 4x4. Split into B1 (2x4) and B2 (2x4).
    # C1 is 1x4.
    #
    # C1 = A11 * B1 + A12 * B2.
    #
    # Now split B1 (2x4) into B11 (2x2) and B12 (2x2).
    # Split B2 (2x4) into B21 (2x2) and B22 (2x2).
    #
    # So:
    # 1. A11 (1x2) * B11 (2x2) -> C11 (1x2) [C cols 0,1]
    # 2. A11 (1x2) * B12 (2x2) -> C12 (1x2) [C cols 2,3]
    # 3. A12 (1x2) * B21 (2x2) -> C13 (1x2) [C cols 0,1]
    # 4. A12 (1x2) * B22 (2x2) -> C14 (1x2) [C cols 2,3]
    
    sub_problems.append({
        'type': 1,
        'a_rows': [0], # A11
        'b_inner': [0, 1], # B11
        'c_rows': [0],
        'c_cols': [2, 3] # C12
    })
    sub_problems.append({
        'type': 1,
        'a_rows': [0], # A12
        'b_inner': [2, 3], # B21
        'c_rows': [0],
        'c_cols': [0, 1] # C13
    })
    sub_problems.append({
        'type': 1,
        'a_rows': [0], # A12
        'b_inner': [2, 3], # B22
        'c_rows': [0],
        'c_cols': [2, 3] # C14
    })
    
    # For A2 (2x4) * B (4x4) -> C2 (2x4)
    # A21 (2x2) * B1 (2x4)
    # A22 (2x2) * B2 (2x4)
    #
    # Split B1 into B11 (2x2) and B12 (2x2)
    # Split B2 into B21 (2x2) and B22 (2x2)
    #
    # 5. A21 (2x2) * B11 (2x2) -> C21 (2x2) [C rows 1,2, C cols 0,1]
    # 6. A21 (2x2) * B12 (2x2) -> C22 (2x2) [C rows 1,2, C cols 2,3]
    # 7. A22 (2x2) * B21 (2x2) -> C23 (2x2) [C rows 1,2, C cols 0,1]
    # 8. A22 (2x2) * B22 (2x2) -> C24 (2x2) [C rows 1,2, C cols 2,3]
    
    sub_problems.append({
        'type': 0,
        'a_rows': [1, 2], # A21
        'b_inner': [0, 1], # B11
        'c_rows': [1, 2],
        'c_cols': [0, 1] # C21
    })
    sub_problems.append({
        'type': 0,
        'a_rows': [1, 2], # A21
        'b_inner': [0, 1], # B12? No, B12 is cols 2,3 of B1?
        # B1 is rows 0,1 of B. B11 is cols 0,1 of B1. B12 is cols 2,3 of B1.
        # So B12 global inner indices are 0,1? No.
        # B is 4x4.
        # B11: rows 0,1; cols 0,1.
        # B12: rows 0,1; cols 2,3.
        # B21: rows 2,3; cols 0,1.
        # B22: rows 2,3; cols 2,3.
        
        # So for sub-problem 6: A21 * B12
        # A21 rows: 1,2.
        # B12 inner (rows of B): 0,1. Cols of B (p-dim): 2,3.
        # Wait, the sub-problem is 2x2x2.
        # The "inner" dimension of the sub-problem corresponds to the rows of B.
        # The "outer" dimension of the sub-problem corresponds to the cols of B.
        # So for B12, the rows are 0,1 and cols are 2,3.
        'b_inner': [0, 1],
        'c_rows': [1, 2],
        'c_cols': [2, 3] # C22
    })
    sub_problems.append({
        'type': 0,
        'a_rows': [1, 2], # A22
        'b_inner': [2, 3], # B21
        'c_rows': [1, 2],
        'c_cols': [0, 1] # C23
    })
    sub_problems.append({
        'type': 0,
        'a_rows': [1, 2], # A22
        'b_inner': [2, 3], # B22
        'c_rows': [1, 2],
        'c_cols': [2, 3] # C24
    })
    
    # Now we need to construct U, V, W for each sub-problem and concatenate.
    
    total_rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    for sp in sub_problems:
        if sp['type'] == 0:
            # 2x2x2 Strassen
            rank_sp = 7
            U_sp, V_sp, W_sp = get_strassen_222()
        else:
            # 1x2x2 Schoolbook
            rank_sp = 4
            U_sp, V_sp, W_sp = get_schoolbook_122()
            
        total_rank += rank_sp
        
        # Map local indices to global indices
        # U_sp is (2, 7) or (2, 4)
        # V_sp is (4, 7) or (4, 4)
        # W_sp is (2, 7) or (2, 4)
        
        U_glob = np.zeros((n*m, rank_sp))
        V_glob = np.zeros((m*p, rank_sp))
        W_glob = np.zeros((p*n, rank_sp))
        
        a_rows = sp['a_rows']
        b_inner = sp['b_inner']
        c_rows = sp['c_rows']
        c_cols = sp['c_cols']
        
        # Local A indices: 0, 1
        # Global A indices: a_rows[0], a_rows[1]
        for local_i in range(len(a_rows)):
            global_i = a_rows[local_i]
            # A is n x m. Index is i*m + j?
            # In the tensor, A index is (i, j).
            # U rows correspond to A entries.
            # U[global_i*m + local_j, r] = U_sp[local_i, local_j, r]?
            # Wait, U_sp is (n_sp, m_sp, rank).
            # In my get_strassen, U is (n_sp, rank).
            # Let's check the shape of U_sp from get_strassen_222.
            # It should be (2, 7).
            # Local j is 0, 1.
            # Global A index is (global_i, local_j).
            # But local_j corresponds to the local m-dimension.
            # The global m-index is not just local_j, it's offset by the split.
            # For A21 (rows 1,2), the cols are 0,1 (global m).
            # For A11 (row 0), the cols are 0,1 (global m).
            
            # So local_j maps to global_m = b_inner[local_j]? No.
            # A is n x m.
            # A21 is rows 1,2; cols 0,1.
            # So local_j=0 maps to global_m=0. local_j=1 maps to global_m=1.
            
            # But what if the sub-problem used cols 2,3?
            # A12 is row 0; cols 2,3.
            # So local_j=0 maps to global_m=2. local_j=1 maps to global_m=3.
            
            # So we need to know the global m-offset for the sub-problem.
            # For sub-problems 1,2: m-offset 0.
            # For sub-problems 3,4: m-offset 2.
            # For sub-problems 5,6: m-offset 0.
            # For sub-problems 7,8: m-offset 2.
            
            # I didn't store m_offset in sp.
            # I can infer it from b_inner?
            # b_inner are the rows of B.
            # The cols of A are the rows of B.
            # So global_m = b_inner[local_j].
            
            for local_j in range(2):
                global_m = b_inner[local_j]
                global_a_idx = global_i * m + global_m
                U_glob[global_a_idx, :] = U_sp[local_i, :]
                
        # V rows correspond to B entries.
        # B is m x p. Index is (j, k).
        # V_sp is (m_sp, p_sp, rank) -> (4, rank) in my helper?
        # No, V_sp is (4, rank).
        # Local V indices: (local_j, local_k).
        # global_j = b_inner[local_j].
        # global_k = c_cols[local_k].
        
        for local_j in range(2):
            for local_k in range(2):
                global_j = b_inner[local_j]
                global_k = c_cols[local_k]
                global_b_idx = global_j * p + global_k
                V_glob[global_b_idx, :] = V_sp[local_j * 2 + local_k, :]
                
        # W rows correspond to C entries.
        # C is n x p. Index is (i, k).
        # W_sp is (n_sp, p_sp, rank) -> (2, rank).
        # Local W indices: (local_i, local_k).
        # global_i = c_rows[local_i].
        # global_k = c_cols[local_k].
        
        for local_i in range(len(c_rows)):
            for local_k in range(2):
                global_i = c_rows[local_i]
                global_k = c_cols[local_k]
                global_c_idx = global_k * n + global_i
                W_glob[global_c_idx, :] = W_sp[local_i * 2 + local_k, :]
                
        U_parts.append(U_glob)
        V_parts.append(V_glob)
        W_parts.append(W_glob)
        
    U_final = np.hstack(U_parts)
    V_final = np.hstack(V_parts)
    W_final = np.hstack(W_parts)
    
    # Convert to lists
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def get_strassen_222():
    # Rank 7
    # Returns U (2,7), V (4,7), W (2,7)
    # Note: The tensor decomposition of matrix multiplication
    # C = A B
    # C_{ik} = sum_j A_{ij} B_{jk}
    
    # Strassen products:
    # M1 = A11 (B11 - B21)
    # M2 = (A11 + A12) B21
    # M3 = (A21 + A22) B12
    # M4 = A22 (B22 - B12)
    # M5 = (A11 + A21) B11
    # M6 = (A12 - A21) (B11 + B22)
    # M7 = (A11 - A12) (B12 + B22)
    
    # U corresponds to A. V to B. W to C.
    # Each column r corresponds to one product.
    # U[:, r] is the vector representing the linear combination of A for product r.
    # V[:, r] is the vector representing the linear combination of B for product r.
    # W[:, r] is the vector representing the linear combination of C contributions for product r.
    
    U = np.zeros((2, 7))
    V = np.zeros((4, 7))
    W = np.zeros((2, 7))
    
    # Indices for A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Indices for B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Indices for C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1: A11 (B11 - B21)
    # A11 is A[0,0]. V: B11 - B21 -> B[0,0] - B[1,0]
    # U[0,0] = 1
    # V[0,0] = 1, V[2,0] = -1
    # W: C11 += M1, C12 += M1? No.
    # C11 = M1 + M2 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 + M4 + M5 - M6
    
    # So W[0,0] = 1 (C11), W[1,0] = 0? No, M1 contributes to C11 and C22.
    # Wait, W is (2,7). Rows are C1 (row 0) and C2 (row 1)?
    # No, W should be (n*p, rank) = (4, 7) in the final global matrix.
    # But in the sub-problem, W is (2, 7)?
    # My previous helper assumed W is (2,7) for 2x2x2?
    # C is 2x2. So 4 entries.
    # So W_sp should be (4, 7).
    
    # Let's fix the helper return shapes.
    # U_sp: (2, 7)
    # V_sp: (4, 7)
    # W_sp: (4, 7)
    
    W = np.zeros((4, 7))
    
    # r=0: M1
    U[0,0] = 1
    V[0,0] = 1
    V[2,0] = -1
    W[0,0] = 1 # C11
    W[3,0] = 1 # C22
    
    # r=1: M2
    U[0,1] = 1
    U[1,1] = 1
    V[2,1] = 1
    W[0,1] = 1 # C11
    W[2,1] = 1 # C21
    
    # r=2: M3
    U[2,2] = 1
    U[3,2] = 1
    V[1,2] = 1
    W[1,2] = 1 # C12
    
    # r=3: M4
    U[3,3] = 1
    V[3,3] = 1
    V[1,3] = -1
    W[2,3] = 1 # C21
    W[3,3] = 1 # C22
    
    # r=4: M5
    U[0,4] = 1
    U[2,4] = 1
    V[0,4] = 1
    W[0,4] = 1 # C11
    W[3,4] = 1 # C22
    
    # r=5: M6
    U[1,5] = 1
    U[2,5] = -1
    V[0,5] = 1
    V[3,5] = 1
    W[3,5] = -1 # C22
    
    # r=6: M7
    U[0,6] = 1
    U[1,6] = -1
    V[1,6] = 1
    V[3,6] = 1
    W[0,6] = -1 # C11
    W[1,6] = 1 # C12
    
    return U, V, W

def get_schoolbook_122():
    # Rank 4
    # A is 1x2. B is 2x2. C is 1x2.
    # U (1,4), V (4,4), W (2,4)
    
    U = np.zeros((1, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    
    # r=0: A0 B00
    U[0,0] = 1
    V[0,0] = 1
    W[0,0] = 1
    
    # r=1: A1 B10
    U[0,1] = 1 # A1 is index 1 of A?
    # A is 1x2. Indices 0,1.
    # U is (1,4). So U[0,1] corresponds to A[0,1]?
    # No, U row index is 0. Column index is the rank.
    # We need to specify which A entry is used.
    # U[local_i, r]. local_i is 0.
    # But we have two A entries: A[0,0] and A[0,1].
    # So U must be (2, 4)?
    # No, A is 1x2. So 2 entries.
    # U should be (2, 4).
    
    U = np.zeros((2, 4))
    
    # r=0: A[0,0] * B[0,0]
    U[0,0] = 1
    V[0,0] = 1
    W[0,0] = 1 # C[0,0]
    
    # r=1: A[0,1] * B[1,0]
    U[1,1] = 1
    V[2,1] = 1
    W[0,1] = 1 # C[0,0]
    
    # r=2: A[0,0] * B[0,1]
    U[0,2] = 1
    V[1,2] = 1
    W[1,2] = 1 # C[0,1]
    
    # r=3: A[0,1] * B[1,1]
    U[1,3] = 1
    V[3,3] = 1
    W[1,3] = 1 # C[0,1]
    
    return U, V, W

