import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition in parts
    # Part 1: A[0:4, 0:4] * B[0:4, 0:4] (4x4x4) using Strassen with inner trivial 2x2
    # Part 2: A[0:4, 0:4] * B[0:4, 4:5] (4x4x1) trivial
    # Part 3: A[0:4, 4:5] * B[4:5, 0:4] (4x1x4) trivial
    # Part 4: A[0:4, 4:5] * B[4:5, 4:5] (4x1x1) trivial
    
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_trivial_2x2(A_block, B_block, C_block):
        # A_block, B_block, C_block are tuples of (row_start, col_start, row_end, col_end)
        # But we need to map to global indices
        pass

    # Helper to add a trivial rank-1 term for specific i, k, j
    def add_term(i, k, j, sign=1):
        nonlocal rank
        u_idx = i * m + k
        v_idx = k * p + j
        w_idx = j * n + i
        
        U_list.append([0.0] * (rank + 1)) # This is inefficient, we should build columns
        # Better to build the matrices directly
        
    # Let's build the matrices as lists of columns (vectors)
    # Then transpose or just return as is if expected shape is (size, rank)
    # The task says U shape (n*m, rank). So U[i, r].
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(i, k, j, coeff=1.0):
        # Adds a term coeff * A[i,k] * B[k,j] to C[i,j]
        # In tensor decomposition:
        # U[i*m+k, r] = coeff
        # V[k*p+j, r] = 1
        # W[j*n+i, r] = 1
        # Wait, the product is U * V * W?
        # The decomposition is sum_r U[i,k,r] V[k,j,r] W[j,i,r] = A[i,k] B[k,j]
        # If we set U=coeff, V=1, W=1, then term is coeff * A[i,k] B[k,j].
        # This is correct.
        
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        u_idx = i * m + k
        v_idx = k * p + j
        w_idx = j * n + i
        
        u_vec[u_idx] = coeff
        v_vec[v_idx] = 1.0
        w_vec[w_idx] = 1.0
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Part 1: 4x4x4 Strassen
    # Indices for 4x4: i in 0..3, k in 0..3, j in 0..3
    # Blocks:
    # A11: i 0,1 k 0,1
    # A12: i 0,1 k 2,3
    # A21: i 2,3 k 0,1
    # A22: i 2,3 k 2,3
    
    # M1 = (A11 + A22)(B11 + B22)
    # Terms:
    # A11*B11: i 0,1 k 0,1 j 0,1
    for i in [0, 1]:
        for k in [0, 1]:
            for j in [0, 1]:
                add_rank1(i, k, j, 1.0)
    # A11*B22: i 0,1 k 0,1 j 2,3
    for i in [0, 1]:
        for k in [0, 1]:
            for j in [2, 3]:
                add_rank1(i, k, j, 1.0)
    # A22*B11: i 2,3 k 2,3 j 0,1
    for i in [2, 3]:
        for k in [2, 3]:
            for j in [0, 1]:
                add_rank1(i, k, j, 1.0)
    # A22*B22: i 2,3 k 2,3 j 2,3
    for i in [2, 3]:
        for k in [2, 3]:
            for j in [2, 3]:
                add_rank1(i, k, j, 1.0)
                
    # M2 = (A21 + A22)B11
    # A21*B11: i 2,3 k 0,1 j 0,1
    for i in [2, 3]:
        for k in [0, 1]:
            for j in [0, 1]:
                add_rank1(i, k, j, 1.0)
    # A22*B11: i 2,3 k 2,3 j 0,1
    for i in [2, 3]:
        for k in [2, 3]:
            for j in [0, 1]:
                add_rank1(i, k, j, 1.0)
                
    # M3 = A11(B12 - B22)
    # A11*B12: i 0,1 k 0,1 j 2,3
    for i in [0, 1]:
        for k in [0, 1]:
            for j in [2, 3]:
                add_rank1(i, k, j, 1.0)
    # -A11*B22: i 0,1 k 0,1 j 2,3
    for i in [0, 1]:
        for k in [0, 1]:
            for j in [2, 3]:
                add_rank1(i, k, j, -1.0)
                
    # M4 = A22(B21 - B11)
    # A22*B21: i 2,3 k 2,3 j 2,3
    for i in [2, 3]:
        for k in [2, 3]:
            for j in [2, 3]:
                add_rank1(i, k, j, 1.0)
    # -A22*B11: i 2,3 k 2,3 j 0,1
    for i in [2, 3]:
        for k in [2, 3]:
            for j in [0, 1]:
                add_rank1(i, k, j, -1.0)
                
    # M5 = (A11 + A12)B22
    # A11*B22: i 0,1 k 0,1 j 2,3
    for i in [0, 1]:
        for k in [0, 1]:
            for j in [2, 3]:
                add_rank1(i, k, j, 1.0)
    # A12*B22: i 0,1 k 2,3 j 2,3
    for i in [0, 1]:
        for k in [2, 3]:
            for j in [2, 3]:
                add_rank1(i, k, j, 1.0)
                
    # M6 = (A21 - A11)(B11 + B12)
    # A21*B11: i 2,3 k 0,1 j 0,1
    for i in [2, 3]:
        for k in [0, 1]:
            for j in [0, 1]:
                add_rank1(i, k, j, 1.0)
    # A21*B12: i 2,3 k 0,1 j 2,3
    for i in [2, 3]:
        for k in [0, 1]:
            for j in [2, 3]:
                add_rank1(i, k, j, 1.0)
    # -A11*B11: i 0,1 k 0,1 j 0,1
    for i in [0, 1]:
        for k in [0, 1]:
            for j in [0, 1]:
                add_rank1(i, k, j, -1.0)
    # -A11*B12: i 0,1 k 0,1 j 2,3
    for i in [0, 1]:
        for k in [0, 1]:
            for j in [2, 3]:
                add_rank1(i, k, j, -1.0)
                
    # M7 = (A12 - A22)(B21 + B22)
    # A12*B21: i 0,1 k 2,3 j 2,3
    for i in [0, 1]:
        for k in [2, 3]:
            for j in [2, 3]:
                add_rank1(i, k, j, 1.0)
    # A12*B22: i 0,1 k 2,3 j 2,3
    for i in [0, 1]:
        for k in [2, 3]:
            for j in [2, 3]:
                add_rank1(i, k, j, 1.0)
    # -A22*B21: i 2,3 k 2,3 j 2,3
    for i in [2, 3]:
        for k in [2, 3]:
            for j in [2, 3]:
                add_rank1(i, k, j, -1.0)
    # -A22*B22: i 2,3 k 2,3 j 2,3
    for i in [2, 3]:
        for k in [2, 3]:
            for j in [2, 3]:
                add_rank1(i, k, j, -1.0)
                
    # Part 2: A[0:4, 0:4] * B[0:4, 4:5] (4x4x1)
    # i 0..3, k 0..3, j 4
    for i in range(4):
        for k in range(4):
            add_rank1(i, k, 4, 1.0)
            
    # Part 3: A[0:4, 4:5] * B[4:5, 0:4] (4x1x4)
    # i 0..3, k 4, j 0..3
    for i in range(4):
        for j in range(4):
            add_rank1(i, 4, j, 1.0)
            
    # Part 4: A[0:4, 4:5] * B[4:5, 4:5] (4x1x1)
    # i 0..3, k 4, j 4
    for i in range(4):
        add_rank1(i, 4, 4, 1.0)
        
    rank = len(U_cols)
    
    # Convert lists of columns to matrix (n*m, rank)
    # U_cols is list of vectors of length n*m
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
