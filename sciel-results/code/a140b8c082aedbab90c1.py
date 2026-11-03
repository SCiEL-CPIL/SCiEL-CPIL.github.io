DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0]*rank for _ in range(n*m)]  # 8 x 50
    V = [[0]*rank for _ in range(m*p)]  # 28 x 50
    W = [[0]*rank for _ in range(p*n)]  # 14 x 50
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
        # a_rows: 2 indices into A's row (0,1)
        # a_cols: 2 indices into A's col (0-3)
        # b_rows: 2 indices into B's row (0-3)
        # b_cols: 2 indices into B's col (0-6)
        # c_rows: 2 indices into C's row (0,1)
        # c_cols: 2 indices into C's col (0-6)
        # A[i, a_cols[j]], B[b_rows[j], b_cols[k]], C[c_rows[i], c_cols[k]]
        
        a00 = a_rows[0]*m + a_cols[0]
        a01 = a_rows[0]*m + a_cols[1]
        a10 = a_rows[1]*m + a_cols[0]
        a11 = a_rows[1]*m + a_cols[1]
        
        b00 = b_rows[0]*p + b_cols[0]
        b01 = b_rows[0]*p + b_cols[1]
        b10 = b_rows[1]*p + b_cols[0]
        b11 = b_rows[1]*p + b_cols[1]
        
        c00 = c_cols[0]*n + c_rows[0]
        c01 = c_cols[1]*n + c_rows[0]
        c10 = c_cols[0]*n + c_rows[1]
        c11 = c_cols[1]*n + c_rows[1]
        
        # m1 = (a00+a11)*(b00+b11)
        col = start_col
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # m2 = (a10+a11)*b00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # m3 = a00*(b01-b11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # m4 = a11*(b10-b00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # m5 = (a00+a01)*b11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # m6 = (a10-a00)*(b00+b01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # m7 = (a01-a11)*(b10+b11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col, start_col):
        # A (2x2) x B (2x1) -> C (2x1)
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        
        a00 = a_rows[0]*m + a_cols[0]
        a01 = a_rows[0]*m + a_cols[1]
        a10 = a_rows[1]*m + a_cols[0]
        a11 = a_rows[1]*m + a_cols[1]
        
        b0 = b_rows[0]*p + b_col
        b1 = b_rows[1]*p + b_col
        
        c0 = c_col*n + c_rows[0]
        c1 = c_col*n + c_rows[1]
        
        col = start_col
        # A[0,0]*B[0] -> C[0]
        U[a00][col] += 1
        V[b0][col] += 1
        W[c0][col] += 1
        col += 1
        
        # A[0,1]*B[1] -> C[0]
        U[a01][col] += 1
        V[b1][col] += 1
        W[c0][col] += 1
        col += 1
        
        # A[1,0]*B[0] -> C[1]
        U[a10][col] += 1
        V[b0][col] += 1
        W[c1][col] += 1
        col += 1
        
        # A[1,1]*B[1] -> C[1]
        U[a11][col] += 1
        V[b1][col] += 1
        W[c1][col] += 1
        col += 1
        
        return col
    
    col = 0
    
    # Block 1a: A[:,0:2] x B[0:2, 0:2] -> C[:,0:2]
    col = add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], col)
    
    # Block 1b: A[:,0:2] x B[0:2, 2:3] -> C[:,2:3]
    col = add_schoolbook_2x2_2x1([0,1], [0,1], [0,1], 2, [0,1], 2, col)
    
    # Block 2a: A[:,2:4] x B[2:4, 0:2] -> C[:,0:2]
    col = add_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], col)
    
    # Block 2b: A[:,2:4] x B[2:4, 2:3] -> C[:,2:3]
    col = add_schoolbook_2x2_2x1([0,1], [2,3], [2,3], 2, [0,1], 2, col)
    
    # Block 3a: A[:,0:2] x B[0:2, 3:5] -> C[:,3:5]
    col = add_strassen([0,1], [0,1], [0,1], [3,4], [0,1], [3,4], col)
    
    # Block 3b: A[:,0:2] x B[0:2, 5:7] -> C[:,5:7]
    col = add_strassen([0,1], [0,1], [0,1], [5,6], [0,1], [5,6], col)
    
    # Block 4a: A[:,2:4] x B[2:4, 3:5] -> C[:,3:5]
    col = add_strassen([0,1], [2,3], [2,3], [3,4], [0,1], [3,4], col)
    
    # Block 4b: A[:,2:4] x B[2:4, 5:7] -> C[:,5:7]
    col = add_strassen([0,1], [2,3], [2,3], [5,6], [0,1], [5,6], col)
    
    return {"U": U, "V": V, "W": W}
