"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W index: k*n + i
    
    u_cols = []
    v_cols = []
    w_cols = []
    
    def add_decomposition(u, v, w, a_row_off, a_col_off, b_row_off, b_col_off):
        """
        Adds a decomposition for a sub-multiplication.
        u: local U (a_rows * a_cols, r)
        v: local V (b_rows * b_cols, r) -- Note: b_rows == a_cols
        w: local W (c_cols * c_rows, r) -- Note: c_cols == b_cols, c_rows == a_rows
        a_row_off: offset in A rows
        a_col_off: offset in A cols (and B rows)
        b_row_off: offset in B rows (should match a_col_off)
        b_col_off: offset in B cols (and C cols)
        """
        a_rows = u.shape[0] // a_cols # This is wrong, u shape is (a_rows * a_cols, r)
        # We need to know the local dimensions to map indices.
        # It's better to pass local n, m, p.
        
    # Let's implement a generic adder that takes local n, m, p and the local U, V, W.
    
    def add_block(local_n, local_m, local_p, a_row_off, a_col_off, b_col_off, u_loc, v_loc, w_loc):
        """
        local_n: rows of A
        local_m: cols of A / rows of B
        local_p: cols of B
        a_row_off: offset for A rows
        a_col_off: offset for A cols (and B rows)
        b_col_off: offset for B cols (and C cols)
        u_loc: shape (local_n * local_m, r)
        v_loc: shape (local_m * local_p, r)
        w_loc: shape (local_p * local_n, r)
        """
        r = u_loc.shape[1]
        for c in range(r):
            # U column
            u_col = [0] * (n * m)
            for i in range(local_n):
                for j in range(local_m):
                    u_col[(a_row_off + i) * m + (a_col_off + j)] = u_loc[i * local_m + j, c]
            
            # V column
            v_col = [0] * (m * p)
            for j in range(local_m):
                for k in range(local_p):
                    v_col[(a_col_off + j) * p + (b_col_off + k)] = v_loc[j * local_p + k, c]
            
            # W column
            # W index is k*n + i
            w_col = [0] * (p * n)
            for k in range(local_p):
                for i in range(local_n):
                    # Local W index: k * local_n + i
                    w_col[(b_col_off + k) * n + (a_row_off + i)] = w_loc[k * local_n + i, c]
            
            u_cols.append(u_col)
            v_cols.append(v_col)
            w_cols.append(w_col)
            
    def get_strassen_222():
        # Returns U, V, W for 2x2x2
        # U: 4x7, V: 4x7, W: 4x7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1 # C11, C22
        
        # M2 = (A21+A22)B11
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1 # C21
        
        # M3 = A11(B12-B22)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1 # C12
        
        # M4 = A22(B21-B11)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1 # C11, C21
        
        # M5 = (A11+A12)B22
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1 # C11, C12
        
        # M6 = (A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1 # C22
        
        # M7 = (A12-A22)(B21+B22)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1; W[3,6]=1 # C11, C22
        
        return U, V, W

    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # 1.1: A(0:2, 0:2) B(0:2, 0:2) -> C(0:2, 0:2)
    u, v, w = get_strassen_222()
    add_block(2, 2, 2, 0, 0, 0, u, v, w)
    
    # 1.2: A(0:2, 2:4) B(2:4, 0:2) -> C(0:2, 0:2)
    u, v, w = get_strassen_222()
    add_block(2, 2, 2, 0, 2, 0, u, v, w)
    
    # 2.1.1: A(0:2, 0:2) B(0:2, 2:4) -> C(0:2, 2:4)
    u, v, w = get_strassen_222()
    add_block(2, 2, 2, 0, 0, 2, u, v, w)
    
    # 2.1.2: A(0:2, 0:2) B(0:2, 4:5) -> C(0:2, 4:5)
    u, v, w = get_schoolbook(2, 2, 1)
    add_block(2, 2, 1, 0, 0, 4, u, v, w)
    
    # 2.2.1: A(0:2, 2:4) B(2:4, 2:4) -> C(0:2, 2:4)
    u, v, w = get_strassen_222()
    add_block(2, 2, 2, 0, 2, 2, u, v, w)
    
    # 2.2.2: A(0:2, 2:4) B(2:4, 4:5) -> C(0:2, 4:5)
    u, v, w = get_schoolbook(2, 2, 1)
    add_block(2, 2, 1, 0, 2, 4, u, v, w)
    
    # 3.1: A(2:3, 0:4) B(0:4, 0:2) -> C(2:3, 0:2)
    u, v, w = get_schoolbook(1, 4, 2)
    add_block(1, 4, 2, 2, 0, 0, u, v, w)
    
    # 3.2: A(2:3, 0:4) B(0:4, 2:5) -> C(2:3, 2:5)
    u, v, w = get_schoolbook(1, 4, 3)
    add_block(1, 4, 3, 2, 0, 2, u, v, w)
    
    U = u_cols
    V = v_cols
    W = w_cols
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
