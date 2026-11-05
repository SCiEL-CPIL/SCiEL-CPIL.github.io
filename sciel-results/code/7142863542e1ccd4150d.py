"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Block decomposition:
    # A is 3x5, split rows into [0:2, 2:3], cols into [0:2, 2:4, 4:5]
    # B is 5x6, split rows into [0:2, 2:4, 4:5], cols into [0:2, 2:4, 4:6]
    # C is 3x6
    
    # We'll build U, V, W matrices column by column (each column is a rank-1 term)
    U_cols = []  # list of (n*m,) vectors
    V_cols = []  # list of (m*p,) vectors
    W_cols = []  # list of (p*n,) vectors
    
    def add_term(u, v, w):
        """Add a rank-1 term. u is (n*m,), v is (m*p,), w is (p*n,)"""
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def make_u(row_idx, col_idx, val=1):
        """Create U vector with val at position row_idx*m + col_idx"""
        vec = [0] * (n * m)
        vec[row_idx * m + col_idx] = val
        return vec
    
    def make_v(row_idx, col_idx, val=1):
        """Create V vector with val at position row_idx*p + col_idx"""
        vec = [0] * (m * p)
        vec[row_idx * p + col_idx] = val
        return vec
    
    def make_w(row_idx, col_idx, val=1):
        """Create W vector with val at position col_idx*n + row_idx (trace convention W[k*n+i,r])"""
        vec = [0] * (p * n)
        vec[col_idx * n + row_idx] = val
        return vec
    
    def add_schoolbook_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """Add schoolbook 8 terms for 2x2 * 2x2 -> 2x2 block"""
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = make_u(a_row_start + i, a_col_start + j)
                    v = make_v(b_row_start + j, b_col_start + k)
                    w = make_w(c_row_start + i, c_col_start + k)
                    add_term(u, v, w)
    
    def add_strassen_2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """Add Strassen 7 terms for 2x2 * 2x2 -> 2x2 block"""
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        # 
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        #
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # Each M is a product of a linear combo of A entries and a linear combo of B entries
        # We need to express each M as a sum of rank-1 terms
        
        # M1 = (A00+A11)*(B00+B11)
        # = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # But we need it as ONE rank-1 term in the tensor decomposition
        # Actually, Strassen works by computing M1 = (A00+A11)*(B00+B11) as a single "multiplication"
        # In the tensor decomposition, this means we have one rank-1 term where:
        # U picks out (A00+A11), V picks out (B00+B11), W contributes to C appropriately
        
        # For M1: U vector has 1 at (0,0) and 1 at (1,1) relative to block
        # V vector has 1 at (0,0) and 1 at (1,1) relative to block
        # W vector: M1 contributes +1 to C00, -1 to C11 (from C00=M1+M4-M5+M7, C11=M1-M2+M3+M6)
        
        # M1 contribution to W: C00 += 1, C11 += 1
        u = [0]*(n*m); u[(a_row_start+0)*m + (a_col_start+0)] = 1; u[(a_row_start+1)*m + (a_col_start+1)] = 1
        v = [0]*(m*p); v[(b_row_start+0)*p + (b_col_start+0)] = 1; v[(b_row_start+1)*p + (b_col_start+1)] = 1
        w = [0]*(p*n); w[(c_col_start+0)*n + (c_row_start+0)] = 1; w[(c_col_start+1)*n + (c_row_start+1)] = 1
        add_term(u, v, w)
        
        # M2 = (A10+A11)*B00
        # C10 += 1, C11 -= 1
        u = [0]*(n*m); u[(a_row_start+1)*m + (a_col_start+0)] = 1; u[(a_row_start+1)*m + (a_col_start+1)] = 1
        v = [0]*(m*p); v[(b_row_start+0)*p + (b_col_start+0)] = 1
        w = [0]*(p*n); w[(c_col_start+0)*n + (c_row_start+1)] = 1; w[(c_col_start+1)*n + (c_row_start+1)] = -1
        add_term(u, v, w)
        
        # M3 = A00*(B01-B11)
        # C01 += 1, C11 += 1
        u = [0]*(n*m); u[(a_row_start+0)*m + (a_col_start+0)] = 1
        v = [0]*(m*p); v[(b_row_start+0)*p + (b_col_start+1)] = 1; v[(b_row_start+1)*p + (b_col_start+1)] = -1
        w = [0]*(p*n); w[(c_col_start+1)*n + (c_row_start+0)] = 1; w[(c_col_start+1)*n + (c_row_start+1)] = 1
        add_term(u, v, w)
        
        # M4 = A11*(B10-B00)
        # C00 += 1, C10 += 1
        u = [0]*(n*m); u[(a_row_start+1)*m + (a_col_start+1)] = 1
        v = [0]*(m*p); v[(b_row_start+1)*p + (b_col_start+0)] = 1; v[(b_row_start+0)*p + (b_col_start+0)] = -1
        w = [0]*(p*n); w[(c_col_start+0)*n + (c_row_start+0)] = 1; w[(c_col_start+0)*n + (c_row_start+1)] = 1
        add_term(u, v, w)
        
        # M5 = (A00+A01)*B11
        # C00 -= 1, C01 += 1
        u = [0]*(n*m); u[(a_row_start+0)*m + (a_col_start+0)] = 1; u[(a_row_start+0)*m + (a_col_start+1)] = 1
        v = [0]*(m*p); v[(b_row_start+1)*p + (b_col_start+1)] = 1
        w = [0]*(p*n); w[(c_col_start+0)*n + (c_row_start+0)] = -1; w[(c_col_start+1)*n + (c_row_start+0)] = 1
        add_term(u, v, w)
        
        # M6 = (A10-A00)*(B00+B10)
        # C11 += 1
        u = [0]*(n*m); u[(a_row_start+1)*m + (a_col_start+0)] = 1; u[(a_row_start+0)*m + (a_col_start+0)] = -1
        v = [0]*(m*p); v[(b_row_start+0)*p + (b_col_start+0)] = 1; v[(b_row_start+1)*p + (b_col_start+0)] = 1
        w = [0]*(p*n); w[(c_col_start+1)*n + (c_row_start+1)] = 1
        add_term(u, v, w)
        
        # M7 = (A01-A11)*(B01+B11)
        # C00 += 1
        u = [0]*(n*m); u[(a_row_start+0)*m + (a_col_start+1)] = 1; u[(a_row_start+1)*m + (a_col_start+1)] = -1
        v = [0]*(m*p); v[(b_row_start+0)*p + (b_col_start+1)] = 1; v[(b_row_start+1)*p + (b_col_start+1)] = 1
        w = [0]*(p*n); w[(c_col_start+0)*n + (c_row_start+0)] = 1
        add_term(u, v, w)
    
    def add_schoolbook_2x1x1x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """2x1 * 1x2 -> 2x2, schoolbook: 4 terms"""
        # A is 2x1 (rows a_row_start to a_row_start+1, col a_col_start)
        # B is 1x2 (row b_row_start, cols b_col_start to b_col_start+1)
        for i in range(2):
            for k in range(2):
                u = make_u(a_row_start + i, a_col_start)
                v = make_v(b_row_start, b_col_start + k)
                w = make_w(c_row_start + i, c_col_start + k)
                add_term(u, v, w)
    
    def add_schoolbook_1x2x2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """1x2 * 2x2 -> 1x2, schoolbook: 4 terms"""
        # A is 1x2 (row a_row_start, cols a_col_start to a_col_start+1)
        # B is 2x2 (rows b_row_start to b_row_start+1, cols b_col_start to b_col_start+1)
        for j in range(2):
            for k in range(2):
                u = make_u(a_row_start, a_col_start + j)
                v = make_v(b_row_start + j, b_col_start + k)
                w = make_w(c_row_start, c_col_start + k)
                add_term(u, v, w)
    
    def add_schoolbook_1x1x1x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """1x1 * 1x2 -> 1x2, schoolbook: 2 terms"""
        u = make_u(a_row_start, a_col_start)
        for k in range(2):
            v = make_v(b_row_start, b_col_start + k)
            w = make_w(c_row_start, c_col_start + k)
            add_term(u, v, w)
    
    # Now build all blocks:
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2] + A[0:2,4:5]*B[4:5,0:2]
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)    # A[0:2,0:2] * B[0:2,0:2]
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)    # A[0:2,2:4] * B[2:4,0:2]
    add_schoolbook_2x1x1x2(0, 4, 4, 0, 0, 0) # A[0:2,4:5] * B[4:5,0:2]
    
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4] + A[0:2,4:5]*B[4:5,2:4]
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)    # A[0:2,0:2] * B[0:2,2:4]
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)    # A[0:2,2:4] * B[2:4,2:4]
    add_schoolbook_2x1x1x2(0, 4, 4, 2, 0, 2) # A[0:2,4:5] * B[4:5,2:4]
    
    # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6] + A[0:2,4:5]*B[4:5,4:6]
    add_strassen_2x2x2(0, 0, 0, 4, 0, 4)    # A[0:2,0:2] * B[0:2,4:6]
    add_strassen_2x2x2(0, 2, 2, 4, 0, 4)    # A[0:2,2:4] * B[2:4,4:6]
    add_schoolbook_2x1x1x2(0, 4, 4, 4, 0, 4) # A[0:2,4:5] * B[4:5,4:6]
    
    # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2] + A[2:3,4:5]*B[4:5,0:2]
    add_schoolbook_1x2x2x2(2, 0, 0, 0, 2, 0) # A[2:3,0:2] * B[0:2,0:2]
    add_schoolbook_1x2x2x2(2, 2, 2, 0, 2, 0) # A[2:3,2:4] * B[2:4,0:2]
    add_schoolbook_1x1x1x2(2, 4, 4, 0, 2, 0) # A[2:3,4:5] * B[4:5,0:2]
    
    # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4] + A[2:3,4:5]*B[4:5,2:4]
    add_schoolbook_1x2x2x2(2, 0, 0, 2, 2, 2) # A[2:3,0:2] * B[0:2,2:4]
    add_schoolbook_1x2x2x2(2, 2, 2, 2, 2, 2) # A[2:3,2:4] * B[2:4,2:4]
    add_schoolbook_1x1x1x2(2, 4, 4, 2, 2, 2) # A[2:3,4:5] * B[4:5,2:4]
    
    # C[2:3, 4:6] = A[2:3,0:2]*B[0:2,4:6] + A[2:3,2:4]*B[2:4,4:6] + A[2:3,4:5]*B[4:5,4:6]
    add_schoolbook_1x2x2x2(2, 0, 0, 4, 2, 4) # A[2:3,0:2] * B[0:2,4:6]
    add_schoolbook_1x2x2x2(2, 2, 2, 4, 2, 4) # A[2:3,2:4] * B[2:4,4:6]
    add_schoolbook_1x1x1x2(2, 4, 4, 4, 2, 4) # A[2:3,4:5] * B[4:5,4:6]
    
    rank = len(U_cols)
    
    # Build final matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            if U_cols[c][i] != 0:
                U[i][c] = U_cols[c][i]
        for i in range(m * p):
            if V_cols[c][i] != 0:
                V[i][c] = V_cols[c][i]
        for i in range(p * n):
            if W_cols[c][i] != 0:
                W[i][c] = W_cols[c][i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
