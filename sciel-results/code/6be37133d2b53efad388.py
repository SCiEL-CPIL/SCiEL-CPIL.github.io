"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=6
    # Splits:
    # n: 2, 1
    # m: 2, 3
    # p: 2, 4 (further 2, 2 for Strassen parts)
    
    # We will build the decomposition block by block.
    # Rank 87 total.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_vals, V_vals, W_vals):
        """
        U_vals: dict mapping (i, j) to coeff
        V_vals: dict mapping (j, k) to coeff
        W_vals: dict mapping (k, i) to coeff
        """
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        
        for (i, j), c in U_vals.items():
            U_row[i * m + j] = c
        for (j, k), c in V_vals.items():
            V_row[j * p + k] = c
        for (k, i), c in W_vals.items():
            W_row[k * n + i] = c
            
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)

    def strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Applies Strassen's algorithm for 2x2x2 multiplication.
        A is 2x2, B is 2x2, C is 2x2.
        A_rows, A_cols, B_rows, B_cols, C_rows, C_cols are lists of indices.
        """
        # A elements: (A_rows[0], A_cols[0]) is a, (A_rows[0], A_cols[1]) is b, etc.
        a = (A_rows[0], A_cols[0])
        b = (A_rows[0], A_cols[1])
        c = (A_rows[1], A_cols[0])
        d = (A_rows[1], A_cols[1])
        
        e = (B_rows[0], B_cols[0])
        f = (B_rows[0], B_cols[1])
        g = (B_rows[1], B_cols[0])
        h = (B_rows[1], B_cols[1])
        
        C00 = (C_rows[0], C_cols[0])
        C01 = (C_rows[0], C_cols[1])
        C10 = (C_rows[1], C_cols[0])
        C11 = (C_rows[1], C_cols[1])
        
        # m1 = (a+d)(e+h)
        add_block({a:1, d:1}, {e:1, h:1}, {C00:1, C11:1})
        # m2 = (c+d)e
        add_block({c:1, d:1}, {e:1}, {C10:1, C11:-1})
        # m3 = a(f-h)
        add_block({a:1}, {f:1, h:-1}, {C01:1, C11:1})
        # m4 = d(g-e)
        add_block({d:1}, {g:1, e:-1}, {C00:1, C10:1})
        # m5 = (a+b)h
        add_block({a:1, b:1}, {h:1}, {C00:-1, C01:1})
        # m6 = (c-a)(e+f)
        add_block({c:1, a:-1}, {e:1, f:1}, {C11:1})
        # m7 = (b-d)(g+h)
        add_block({b:1, d:-1}, {g:1, h:1}, {C00:1})

    def schoolbook_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Schoolbook multiplication for rectangular blocks.
        A is len(A_rows) x len(A_cols)
        B is len(B_rows) x len(B_cols)
        C is len(C_rows) x len(C_cols)
        len(A_cols) must equal len(B_rows).
        """
        for r in range(len(C_rows)):
            for s in range(len(C_cols)):
                for t in range(len(A_cols)):
                    a_idx = (A_rows[r], A_cols[t])
                    b_idx = (B_rows[t], B_cols[s])
                    c_idx = (C_rows[r], C_cols[s])
                    add_block({a_idx:1}, {b_idx:1}, {c_idx:1})

    # Block 1: C11 (2x2) from A11(2x2)B11(2x2) + A12(2x3)B21(3x2)
    # A11: rows 0,1; cols 0,1
    # B11: rows 0,1; cols 0,1
    # C11: rows 0,1; cols 0,1
    strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # A12: rows 0,1; cols 2,3,4
    # B21: rows 2,3,4; cols 0,1
    # C11: rows 0,1; cols 0,1
    schoolbook_block([0,1], [2,3,4], [2,3,4], [0,1], [0,1], [0,1])
    
    # Block 2: C12 (2x4) from A11(2x2)B12(2x4) + A12(2x3)B22(3x4)
    # B12: rows 0,1; cols 2,3,4,5
    # B22: rows 2,3,4; cols 2,3,4,5
    # C12: rows 0,1; cols 2,3,4,5
    
    # A11 B12 part:
    # Split B12 into two 2x2 blocks: cols 2,3 and 4,5
    # B12a: rows 0,1; cols 2,3 -> C12a: rows 0,1; cols 2,3
    strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # B12b: rows 0,1; cols 4,5 -> C12b: rows 0,1; cols 4,5
    strassen_block([0,1], [0,1], [0,1], [4,5], [0,1], [4,5])
    
    # A12 B22 part:
    # Split B22 into two 3x2 blocks: cols 2,3 and 4,5
    # B22a: rows 2,3,4; cols 2,3 -> C12a: rows 0,1; cols 2,3
    schoolbook_block([0,1], [2,3,4], [2,3,4], [2,3], [0,1], [2,3])
    # B22b: rows 2,3,4; cols 4,5 -> C12b: rows 0,1; cols 4,5
    schoolbook_block([0,1], [2,3,4], [2,3,4], [4,5], [0,1], [4,5])
    
    # Block 3: C21 (1x2) from A21(1x2)B11(2x2) + A22(1x3)B21(3x2)
    # A21: row 2; cols 0,1
    # B11: rows 0,1; cols 0,1
    # C21: row 2; cols 0,1
    schoolbook_block([2], [0,1], [0,1], [0,1], [2], [0,1])
    
    # A22: row 2; cols 2,3,4
    # B21: rows 2,3,4; cols 0,1
    # C21: row 2; cols 0,1
    schoolbook_block([2], [2,3,4], [2,3,4], [0,1], [2], [0,1])
    
    # Block 4: C22 (1x4) from A21(1x2)B12(2x4) + A22(1x3)B22(3x4)
    # A21: row 2; cols 0,1
    # B12: rows 0,1; cols 2,3,4,5
    # C22: row 2; cols 2,3,4,5
    schoolbook_block([2], [0,1], [0,1], [2,3,4,5], [2], [2,3,4,5])
    
    # A22: row 2; cols 2,3,4
    # B22: rows 2,3,4; cols 2,3,4,5
    # C22: row 2; cols 2,3,4,5
    schoolbook_block([2], [2,3,4], [2,3,4], [2,3,4,5], [2], [2,3,4,5])

    # Convert lists of columns to matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    rank = len(U_cols)
    U = [col for col in U_cols]
    V = [col for col in V_cols]
    W = [col for col in W_cols]
    
    # Transpose to row-major format as required by evaluator?
    # "U', 'V', 'W': factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # My U_cols is a list of rows (each row is a vector of length n*m).
    # So U is already (rank, n*m). I need (n*m, rank).
    
    U_final = [[0]*rank for _ in range(n*m)]
    V_final = [[0]*rank for _ in range(m*p)]
    W_final = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        for i in range(n*m):
            U_final[i][r] = U[r][i]
        for j in range(m*p):
            V_final[j][r] = V[r][j]
        for k in range(p*n):
            W_final[k][r] = W[r][k]
            
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
