"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication using Strassen.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    # Strassen's 7 products for 2x2 matrix mult
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # U: A elements indexed as A[0]=a, A[1]=b, A[2]=c, A[3]=d
    # V: B elements indexed as B[0]=e, B[1]=f, B[2]=g, B[3]=h
    # W: C elements indexed as C[0]=c00, C[1]=c01, C[2]=c10, C[3]=c11
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c00 gets +m1, c11 gets +m1... wait
    
    # Let me be more careful. W[k*2+i, t] means W index for C[i,j] is j*2+i
    # C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]
    
    # m1 = (a+d)(e+h) contributes to c00 (+), c11 (+), c01 (-), c10 (-)
    # Wait: C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    # c00 = m1+m4-m5+m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # t=0: m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1   # c00
    W[3][0] = 1   # c11
    
    # t=1: m2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1   # c10
    W[3][1] = -1  # c11
    
    # t=2: m3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1   # c01
    W[3][2] = 1   # c11
    
    # t=3: m4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1   # c00
    W[2][3] = 1   # c10
    
    # t=4: m5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # c00
    W[1][4] = 1   # c01
    
    # t=5: m6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # c11
    
    # t=6: m7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # c00
    
    return U, V, W


def _schoolbook_block(ra, ca, rb, cb, rc, cc):
    """
    Schoolbook decomposition for ra x ca times rb x cb -> rc x cc.
    Returns (U, V, W) with shapes (ra*ca, rank), (rb*cb, rank), (rc*cc, rank)
    where rank = ra*ca*rb (assuming ca==rb and rc==ra, cc==cb for our use cases)
    Actually for our blocks: A is (ra x ca), B is (ca x cb), C is (ra x cb)
    """
    rank = ra * ca * cb
    U = [[0]*rank for _ in range(ra*ca)]
    V = [[0]*rank for _ in range(ca*cb)]
    W = [[0]*rank for _ in range(ra*cb)]
    
    col = 0
    for i in range(ra):
        for j in range(ca):
            for k in range(cb):
                U[i*ca + j][col] = 1
                V[j*cb + k][col] = 1
                W[k*ra + i][col] = 1
                col += 1
    
    return U, V, W


def _block_product_UVW(U1, V1, W1, U2, V2, W2, offset):
    """
    Concatenate two decompositions along the rank dimension.
    U1, U2 have same number of rows, etc.
    """
    r1 = len(V1[0]) if V1 else 0
    r2 = len(V2[0]) if V2 else 0
    rank = r1 + r2
    nU = len(U1)
    nV = len(V1)
    nW = len(W1)
    
    U = [[0]*rank for _ in range(nU)]
    V = [[0]*rank for _ in range(nV)]
    W = [[0]*rank for _ in range(nW)]
    
    for i in range(nU):
        for j in range(r1):
            U[i][j] = U1[i][j]
        for j in range(r2):
            U[i][r1+j] = U2[i][j]
    
    for i in range(nV):
        for j in range(r1):
            V[i][j] = V1[i][j]
        for j in range(r2):
            V[i][r1+j] = V2[i][j]
    
    for i in range(nW):
        for j in range(r1):
            W[i][j] = W1[i][j]
        for j in range(r2):
            W[i][r1+j] = W2[i][j]
    
    return U, V, W


def construct():
    """
    Matrix multiplication tensor decomposition for 4x5 * 5x5 -> 4x5.
    Uses block decomposition with Strassen for 2x2x2 blocks.
    Total rank: 98
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Block structure:
    # A (4x5) split into rows [2,2], cols [2,3]
    # B (5x5) split into rows [2,3], cols [2,3]
    # C (4x5) split into rows [2,2], cols [2,3]
    
    # Products needed:
    # C_{00} = A_{00}B_{00} + A_{01}B_{10}
    # C_{01} = A_{00}B_{01} + A_{01}B_{11}
    # C_{10} = A_{10}B_{00} + A_{11}B_{10}
    # C_{11} = A_{10}B_{01} + A_{11}B_{11}
    
    # We need to build the full U, V, W matrices.
    # U has shape (4*5, rank) = (20, rank)
    # V has shape (5*5, rank) = (25, rank)
    # W has shape (5*4, rank) = (20, rank)
    
    # Let's collect all terms and then assemble.
    # Each term is (u_dict, v_dict, w_dict) where keys are indices.
    
    terms = []  # list of (u_entries, v_entries, w_entries)
    
    # Helper to add a schoolbook block product
    def add_schoolbook(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        A is at rows A_rows, cols A_cols in the full A matrix
        B is at rows B_rows, cols B_cols in the full B matrix  
        C is at rows C_rows, cols C_cols in the full C matrix
        """
        ra = len(A_rows)
        ca = len(A_cols)
        rb = len(B_rows)
        cb = len(B_cols)
        rc = len(C_rows)
        cc = len(C_cols)
        
        for i in range(ra):
            for j in range(ca):
                for k in range(cb):
                    u_idx = A_rows[i] * m + A_cols[j]
                    v_idx = B_rows[j] * p + B_cols[k]
                    w_idx = C_cols[k] * n + C_rows[i]
                    terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
    
    # Helper to add Strassen 2x2x2
    def add_strassen(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        For 2x2 blocks: A is 2x2, B is 2x2, C is 2x2
        A_rows, A_cols, B_rows, B_cols, C_rows, C_cols are lists of length 2
        """
        U_s, V_s, W_s = _strassen_2x2x2()
        for t in range(7):
            u_entries = []
            v_entries = []
            w_entries = []
            for i in range(4):
                if U_s[i][t] != 0:
                    u_entries.append((A_rows[i//2] * m + A_cols[i%2], U_s[i][t]))
            for i in range(4):
                if V_s[i][t] != 0:
                    v_entries.append((B_rows[i//2] * p + B_cols[i%2], V_s[i][t]))
            for i in range(4):
                if W_s[i][t] != 0:
                    w_entries.append((C_cols[i%2] * n + C_rows[i//2], W_s[i][t]))
            terms.append((u_entries, v_entries, w_entries))
    
    # Block definitions:
    # A rows: [0,1] for block 0, [2,3] for block 1
    # A cols: [0,1] for block 0, [2,3,4] for block 1
    # B rows: [0,1] for block 0, [2,3,4] for block 1
    # B cols: [0,1] for block 0, [2,3,4] for block 1
    # C rows: [0,1] for block 0, [2,3] for block 1
    # C cols: [0,1] for block 0, [2,3,4] for block 1
    
    A_row_blocks = [[0,1], [2,3]]
    A_col_blocks = [[0,1], [2,3,4]]
    B_row_blocks = [[0,1], [2,3,4]]
    B_col_blocks = [[0,1], [2,3,4]]
    C_row_blocks = [[0,1], [2,3]]
    C_col_blocks = [[0,1], [2,3,4]]
    
    # C_{00} = A_{00}B_{00} + A_{01}B_{10}
    # A_{00}: 2x2, B_{00}: 2x2 -> Strassen
    add_strassen(A_row_blocks[0], A_col_blocks[0], B_row_blocks[0], B_col_blocks[0], C_row_blocks[0], C_col_blocks[0])
    # A_{01}: 2x3, B_{10}: 3x2 -> schoolbook
    add_schoolbook(A_row_blocks[0], A_col_blocks[1], B_row_blocks[1], B_col_blocks[0], C_row_blocks[0], C_col_blocks[0])
    
    # C_{01} = A_{00}B_{01} + A_{01}B_{11}
    # A_{00}: 2x2, B_{01}: 2x3 -> schoolbook
    add_schoolbook(A_row_blocks[0], A_col_blocks[0], B_row_blocks[0], B_col_blocks[1], C_row_blocks[0], C_col_blocks[1])
    # A_{01}: 2x3, B_{11}: 3x3 -> schoolbook
    add_schoolbook(A_row_blocks[0], A_col_blocks[1], B_row_blocks[1], B_col_blocks[1], C_row_blocks[0], C_col_blocks[1])
    
    # C_{10} = A_{10}B_{00} + A_{11}B_{10}
    # A_{10}: 2x2, B_{00}: 2x2 -> Strassen
    add_strassen(A_row_blocks[1], A_col_blocks[0], B_row_blocks[0], B_col_blocks[0], C_row_blocks[1], C_col_blocks[0])
    # A_{11}: 2x3, B_{10}: 3x2 -> schoolbook
    add_schoolbook(A_row_blocks[1], A_col_blocks[1], B_row_blocks[1], B_col_blocks[0], C_row_blocks[1], C_col_blocks[0])
    
    # C_{11} = A_{10}B_{01} + A_{11}B_{11}
    # A_{10}: 2x2, B_{01}: 2x3 -> schoolbook
    add_schoolbook(A_row_blocks[1], A_col_blocks[0], B_row_blocks[0], B_col_blocks[1], C_row_blocks[1], C_col_blocks[1])
    # A_{11}: 2x3, B_{11}: 3x3 -> schoolbook
    add_schoolbook(A_row_blocks[1], A_col_blocks[1], B_row_blocks[1], B_col_blocks[1], C_row_blocks[1], C_col_blocks[1])
    
    # Now assemble U, V, W
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for t, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][t] = val
        for idx, val in v_entries:
            V[idx][t] = val
        for idx, val in w_entries:
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
