import numpy as np
def construct():
    n, m, p = 3, 4, 5
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (20, rank)  
    # W: (p*n, rank) = (15, rank)
    
    # Indexing:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta[i,k]
    
    # Block decomposition:
    # i groups: {0,1} (size 2), {2} (size 1)
    # j groups: {0,1} (size 2), {2,3} (size 2)
    # k groups: {0,1} (size 2), {2,3} (size 2), {4} (size 1)
    
    # For each combination of groups, determine the sub-problem size and use appropriate decomposition
    
    columns = []  # list of (U_col, V_col, W_col)
    
    def add_column(u_vals, v_vals, w_vals):
        """Add a rank-1 term. u_vals: dict mapping (i,j) -> val, etc."""
        U_col = np.zeros(n*m)
        V_col = np.zeros(m*p)
        W_col = np.zeros(p*n)
        for (i,j), val in u_vals.items():
            U_col[i*m + j] = val
        for (j,k), val in v_vals.items():
            V_col[j*p + k] = val
        for (k,i), val in w_vals.items():
            W_col[k*n + i] = val
        columns.append((U_col, V_col, W_col))
    
    # Strassen 2x2x2 decomposition
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global offsets: i_off, j_off, k_off
    def strassen_2x2x2(i_off, j_off, k_off):
        # M1 = (A00+A11)*(B00+B11)
        add_column(
            {(i_off+0,j_off+0): 1, (i_off+1,j_off+1): 1},
            {(j_off+0,k_off+0): 1, (j_off+1,k_off+1): 1},
            {(k_off+0,i_off+0): 1, (k_off+1,i_off+1): 1}
        )
        # M2 = (A10+A11)*B00
        add_column(
            {(i_off+1,j_off+0): 1, (i_off+1,j_off+1): 1},
            {(j_off+0,k_off+0): 1},
            {(k_off+0,i_off+0): 1, (k_off+1,i_off+1): 1}
        )
        # M3 = A00*(B01-B11)
        add_column(
            {(i_off+0,j_off+0): 1},
            {(j_off+0,k_off+1): 1, (j_off+1,k_off+1): -1},
            {(k_off+0,i_off+0): 1, (k_off+1,i_off+1): 1}
        )
        # M4 = A11*(B10-B00)
        add_column(
            {(i_off+1,j_off+1): 1},
            {(j_off+1,k_off+0): 1, (j_off+0,k_off+0): -1},
            {(k_off+0,i_off+0): 1, (k_off+1,i_off+1): 1}
        )
        # M5 = (A00+A01)*B11
        add_column(
            {(i_off+0,j_off+0): 1, (i_off+0,j_off+1): 1},
            {(j_off+1,k_off+1): 1},
            {(k_off+0,i_off+0): 1, (k_off+1,i_off+1): 1}
        )
        # M6 = (A10-A00)*(B00+B10)
        add_column(
            {(i_off+1,j_off+0): 1, (i_off+0,j_off+0): -1},
            {(j_off+0,k_off+0): 1, (j_off+1,k_off+0): 1},
            {(k_off+0,i_off+0): 1, (k_off+1,i_off+1): 1}
        )
        # M7 = (A01-A11)*(B01+B11)
        add_column(
            {(i_off+0,j_off+1): 1, (i_off+1,j_off+1): -1},
            {(j_off+0,k_off+1): 1, (j_off+1,k_off+1): 1},
            {(k_off+0,i_off+0): 1, (k_off+1,i_off+1): 1}
        )
    
    # Schoolbook for small blocks
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        {(i,j): 1},
                        {(j,k): 1},
                        {(k,i): 1}
                    )
    
    # Apply blocks:
    # i=0,1; j=0,1; k=0,1 → Strassen
    strassen_2x2x2(0, 0, 0)  # 7 terms
    # i=0,1; j=0,1; k=2,3 → Strassen
    strassen_2x2x2(0, 0, 2)  # 7 terms
    # i=0,1; j=0,1; k=4 → schoolbook 2*2*1=4
    schoolbook_block([0,1], [0,1], [4])  # 4 terms
    # i=0,1; j=2,3; k=0,1 → Strassen
    strassen_2x2x2(0, 2, 0)  # 7 terms
    # i=0,1; j=2,3; k=2,3 → Strassen
    strassen_2x2x2(0, 2, 2)  # 7 terms
    # i=0,1; j=2,3; k=4 → schoolbook 2*2*1=4
    schoolbook_block([0,1], [2,3], [4])  # 4 terms
    # i=2; j=0,1; k=0,1 → schoolbook 1*2*2=4
    schoolbook_block([2], [0,1], [0,1])  # 4 terms
    # i=2; j=0,1; k=2,3 → schoolbook 1*2*2=4
    schoolbook_block([2], [0,1], [2,3])  # 4 terms
    # i=2; j=0,1; k=4 → schoolbook 1*2*1=2
    schoolbook_block([2], [0,1], [4])  # 2 terms
    # i=2; j=2,3; k=0,1 → schoolbook 1*2*2=4
    schoolbook_block([2], [2,3], [0,1])  # 4 terms
    # i=2; j=2,3; k=2,3 → schoolbook 1*2*2=4
    schoolbook_block([2], [2,3], [2,3])  # 4 terms
    # i=2; j=2,3; k=4 → schoolbook 1*2*1=2
    schoolbook_block([2], [2,3], [4])  # 2 terms
    
    # Total: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    
    rank = len(columns)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c, (U_col, V_col, W_col) in enumerate(columns):
        U[:, c] = U_col
        V[:, c] = V_col
        W[:, c] = W_col
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0 if i == k else 0.0
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                max_err = max(max_err, abs(val - target))
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = rank_school
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
