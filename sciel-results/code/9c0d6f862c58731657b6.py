"""Matrix multiplication 4x4x5 with rank 72 via recursive Strassen."""

def _strassen_222():
    """Return (U, V, W) for M_{2,2,2} with rank 7, 1-based local indices."""
    # Each term: U[row, c], V[col, c], W[prod, c] where row=i*m+j (m=2), col=j*p+k (p=2), prod=k*n+i (n=2)
    # Standard Strassen 7 products for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # Local indices: i,j,k in {0,1}
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    # e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    #
    # U entries (i,j):
    # (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # V entries (j,k):
    # (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # W entries (k,i):
    # (0,0)=C11, (0,1)=C21, (1,0)=C12, (1,1)=C22
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1  # a, d
    V[0][0] = 1; V[3][0] = 1  # e, h
    W[0][0] = 1; W[3][0] = 1  # C11, C22
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1  # c, d
    V[0][1] = 1               # e
    W[1][0] = 1; W[3][0] = 1  # C21, C22
    
    # M3 = a(f-h)
    U[0][2] = 1               # a
    V[1][2] = 1; V[3][2] = -1 # f, -h
    W[2][0] = 1; W[3][0] = 1  # C12, C22
    
    # M4 = d(g-e)
    U[3][3] = 1               # d
    V[2][3] = 1; V[0][3] = -1 # g, -e
    W[0][0] = 1; W[1][0] = 1  # C11, C21
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1  # a, b
    V[3][4] = 1               # h
    W[0][0] = -1; W[2][0] = 1 # -C11, C12
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1 # c, -a
    V[0][5] = 1; V[1][5] = 1  # e, f
    W[3][0] = 1               # C22
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1 # b, -d
    V[2][6] = 1; V[3][6] = 1  # g, h
    W[0][0] = 1               # C11
    
    return U, V, W

def _block_combine(n1, n2, m1, m2, p1, p2, U1, V1, W1, U2, V2, W2):
    """Combine two block decompositions into one.
    Block 1: (n1 x m1) @ (m1 x p1) -> (n1 x p1)
    Block 2: (n2 x m2) @ (m2 x p2) -> (n2 x p2)
    Total: (n x m) @ (m x p) -> (n x p) where n=n1+n2, m=m1+m2, p=p1+p2
    
    The full matrix:
    A = [[A1, A2],     B = [[B1, B3],     C = [[C1, C3],
         [A3, A4]]       [B2, B4]]          [C2, C4]]
    
    C1 = A1*B1, C2 = A3*B1, C3 = A1*B3, C4 = A3*B3
    
    U shape: (n*m) x rank = ((n1+n2)*(m1+m2)) x rank
    V shape: (m*p) x rank = ((m1+m2)*(p1+p2)) x rank
    W shape: (p*n) x rank = ((p1+p2)*(n1+n2)) x rank
    
    U index: global (i,j) -> i*(m1+m2)+j
    For block 1: i in [0,n1), j in [0,m1) -> local i*m1+j
    For block 3: i in [n1,n1+n2), j in [0,m1) -> local (i-n1)*m2+j... wait
    Actually A2 is n1 x m2, A3 is n2 x m1, A4 is n2 x m2.
    
    Let me use offsets. U1 is for (n1 x m1) part: rows 0..n1-1, cols 0..m1-1 of A
    U2 is for (n2 x m2) part: rows n1..n1+n2-1, cols m1..m1+m2-1 of A
    But we also need cross terms? No, each sub-problem is independent.
    
    Wait, C1 = A1*B1: uses A1 (n1 x m1) and B1 (m1 x p1)
    C2 = A3*B1: uses A3 (n2 x m1) and B1 (m1 x p1)
    C3 = A1*B3: uses A1 (n1 x m1) and B3 (m1 x p2)
    C4 = A3*B3: uses A3 (n2 x m1) and B3 (m1 x p2)
    
    So we need 4 sub-problems:
    1. (n1 x m1) @ (m1 x p1) -> C1
    2. (n2 x m1) @ (m1 x p1) -> C2
    3. (n1 x m1) @ (m1 x p2) -> C3
    4. (n2 x m1) @ (m1 x p2) -> C4
    
    But U1, V1, W1 is for (n1 x m1) @ (m1 x p1).
    We need separate decompositions for each.
    """
    pass

def _make_decomp(n, m, p, U_local, V_local, W_local, row_offset, col_offset, out_row_offset, out_col_offset, rank):
    """Place a local decomposition into a global one.
    U_local: (n*m) x rank_sub
    V_local: (m*p) x rank_sub
    W_local: (p*n) x rank_sub
    row_offset: offset in U rows (i offset * m + j offset)
    Actually, we need to map local (i,j) to global (i+row_off, j+col_off)
    Local U index: i*m+j
    Global U index: (i+row_off)*m_global + (j+col_off)
    
    This is getting complex. Let me just build the full thing directly.
    """
    pass

def construct():
    n, m, p = 4, 4, 5
    
    # Decompose 4x4x5 into 2x4x5 + 2x4x5
    # 2x4x5 = 2x4x2 + 2x4x2 + 2x4x1
    # 2x4x2 = 2x2x2 + 2x2x2 (split m=4 into two m=2)
    # So 2x4x2 = rank 7 + rank 7 = 14
    # 2x4x1 = rank 8 (schoolbook: 2*4*1=8)
    # 2x4x5 = 14 + 14 + 8 = 36
    # 4x4x5 = 36 + 36 = 72
    
    rank = 72
    U = [[0]*rank for _ in range(n*m)]  # 16 x 72
    V = [[0]*rank for _ in range(m*p)]  # 20 x 72
    W = [[0]*rank for _ in range(p*n)]  # 20 x 72
    
    def _place_222(U_src, V_src, W_src, i_off, j_off, k_off, base_col):
        """Place a 2x2x2 Strassen decomposition at global offsets.
        i_off: row offset in A (0 or 2 for top/bottom half of n=4)
        j_off: col offset in A (0, 2 for left/right half of m=4)
        k_off: row offset in B (0, 2 for top/bottom half of p=5... wait p=5)
        
        Actually for 2x2x2: n_sub=2, m_sub=2, p_sub=2
        Global i = i_off + i_local, where i_local in {0,1}
        Global j = j_off + j_local, where j_local in {0,1}
        Global k = k_off + k_local, where k_local in {0,1}
        
        U index: global_i * m + global_j = (i_off+i_loc)*4 + (j_off+j_loc)
        V index: global_j * p + global_k = (j_off+j_loc)*5 + (k_off+k_loc)
        W index: global_k * n + global_i = (k_off+k_loc)*4 + (i_off+i_loc)
        """
        for c in range(7):
            col = base_col + c
            # U
            for li in range(2):
                for lj in range(2):
                    val = U_src[li*2+lj][c]
                    if val != 0:
                        gi = i_off + li
                        gj = j_off + lj
                        U[gi*4 + gj][col] = val
            # V
            for lj in range(2):
                for lk in range(2):
                    val = V_src[lj*2+lk][c]
                    if val != 0:
                        gj = j_off + lj
                        gk = k_off + lk
                        V[gj*5 + gk][col] = val
            # W
            for lk in range(2):
                for li in range(2):
                    val = W_src[lk*2+li][c]
                    if val != 0:
                        gk = k_off + lk
                        gi = i_off + li
                        W[gk*4 + gi][col] = val
    
    U222, V222, W222 = _strassen_222()
    
    def _place_schoolbook_nm1(n_sub, m_sub, i_off, j_off, k_off, base_col):
        """Place schoolbook for (n_sub x m_sub) @ (m_sub x 1).
        rank = n_sub * m_sub
        U[gi*m+gj][col] = 1 for each (gi,gj)
        V[gj*5+gk][col] = 1
        W[gk*4+gi][col] = 1
        """
        for li in range(n_sub):
            for lj in range(m_sub):
                col = base_col + li*m_sub + lj
                gi = i_off + li
                gj = j_off + lj
                gk = k_off
                U[gi*4 + gj][col] = 1
                V[gj*5 + gk][col] = 1
                W[gk*4 + gi][col] = 1
    
    # Now build the full 4x4x5
    # Split into top (n=2) and bottom (n=2)
    # Top: i_off=0, bottom: i_off=2
    
    col = 0
    # For each n-half (top, bottom):
    for i_half in range(2):
        i_off = i_half * 2
        # For each m-half (left m=2, right m=2) within 2x4:
        # 2x4x5 = (2x2x5) + (2x2x5) where first is j_off=0, second is j_off=2
        for j_half in range(2):
            j_off = j_half * 2
            # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1
            # 2x2x2: k_off=0, k_off=2
            for k_half in range(2):
                k_off = k_half * 2
                _place_222(U222, V222, W222, i_off, j_off, k_off, col)
                col += 7
            # 2x2x1: k_off=4
            _place_schoolbook_nm1(2, 2, i_off, j_off, 4, col)
            col += 4
    
    # col should be 72
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
