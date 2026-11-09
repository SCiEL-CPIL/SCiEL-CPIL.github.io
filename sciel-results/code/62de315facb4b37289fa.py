import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (15, rank)
    # W: (p*n, rank) = (10, rank)
    
    # We will build a list of terms. Each term is a dict of non-zeros.
    # Or just accumulate into lists of lists.
    # Since rank is small (28), we can just append columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        u_dict: { (i, j): coeff }
        v_dict: { (j, k): coeff }
        w_dict: { (k, i): coeff }
        """
        u_vec = [0.0] * (n * m)
        for (i, j), val in u_dict.items():
            u_vec[i * m + j] = val
            
        v_vec = [0.0] * (m * p)
        for (j, k), val in v_dict.items():
            v_vec[j * p + k] = val
            
        w_vec = [0.0] * (p * n)
        for (k, i), val in w_dict.items():
            w_vec[k * n + i] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Strassen terms for 2x2x2
    # Indices: A: (0,0),(0,1),(1,0),(1,1) -> i in 0,1; j in 0,1
    #          B: (0,0),(0,1),(1,0),(1,1) -> j in 0,1; k in 0,1
    #          C: (0,0),(0,1),(1,0),(1,1) -> k in 0,1; i in 0,1
    
    strassen_terms = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(0,0):-1, (1,0):1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    # Helper to add Strassen block
    def add_strassen_block(a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_k_offset, c_i_offset):
        # a_i_offset, a_j_offset are 0 because we are using the first 2 rows/cols of A_L
        # Actually, A_L is rows 0,1 and cols 0,1.
        # B_L is rows 0,1 and cols k_offset, k_offset+1.
        # C is cols k_offset, k_offset+1 and rows 0,1.
        
        for u, v, w in strassen_terms:
            new_u = {}
            for (i, j), val in u.items():
                # i is 0 or 1, j is 0 or 1
                # Global A index: i * m + (j + a_j_offset)
                new_u[(i, j + a_j_offset)] = val
                
            new_v = {}
            for (j, k), val in v.items():
                # j is 0 or 1, k is 0 or 1
                # Global B index: (j + b_j_offset) * p + (k + b_k_offset)
                new_v[(j + b_j_offset, k + b_k_offset)] = val
                
            new_w = {}
            for (k, i), val in w.items():
                # k is 0 or 1, i is 0 or 1
                # Global C index: (k + c_k_offset) * n + (i + c_i_offset)
                new_w[(k + c_k_offset, i + c_i_offset)] = val
                
            add_term(new_u, new_v, new_w)
            
    def add_schoolbook_block(a_i_range, a_j_range, b_j_range, b_k_range, c_k_range, c_i_range):
        for i in a_i_range:
            for j in a_j_range:
                for k in b_k_range:
                    # Term for A[i,j] * B[j,k] -> C[k,i]
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    add_term(u, v, w)

    # Block 1: C cols 0,1
    # A_L (cols 0,1) * B_1_L (rows 0,1, cols 0,1)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # A_R (col 2) * B_1_R (row 2, cols 0,1)
    # i in 0,1; j=2; k in 0,1
    add_schoolbook_block([0, 1], [2], [2], [0, 1], [0, 1], [0, 1])

    # Block 2: C cols 2,3
    # A_L (cols 0,1) * B_2_L (rows 0,1, cols 2,3)
    add_strassen_block(0, 0, 0, 2, 2, 0)
    # A_R (col 2) * B_2_R (row 2, cols 2,3)
    add_schoolbook_block([0, 1], [2], [2], [2, 3], [2, 3], [0, 1])

    # Block 3: C col 4
    # A_L (cols 0,1) * B_3_L (rows 0,1, col 4)
    add_schoolbook_block([0, 1], [0, 1], [0, 1], [4], [4], [0, 1])
    # A_R (col 2) * B_3_R (row 2, col 4)
    add_schoolbook_block([0, 1], [2], [2], [4], [4], [0, 1])

    rank = len(U_cols)
    # Transpose to shape (size, rank)
    U = [[U_cols[c][i] for c in range(rank)] for i in range(n * m)]
    V = [[V_cols[c][i] for c in range(rank)] for i in range(m * p)]
    W = [[W_cols[c][i] for c in range(rank)] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
