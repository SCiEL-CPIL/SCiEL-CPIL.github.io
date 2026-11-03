import time

START = time.time()
DEADLINE = 140.0

def _check_time():
    return time.time() >= DEADLINE

def _strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen.
    U: 4x7, V: 4x7, W: 4x7.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    r = 7
    U = [[0]*r for _ in range(4)]
    V = [[0]*r for _ in range(4)]
    W = [[0]*r for _ in range(4)]
    
    # p0 = a*e
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # p1 = c*f
    U[2][1] = 1; V[1][1] = 1; W[2][1] = 1 # W index k=0, i=1 -> 0*2+1=1? No.
    # W index is k*2+i.
    # C10 is k=0, i=1. Index 1. Correct.
    
    # p2 = (a+b)*(f-h)
    U[0][2] = 1; U[1][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C01 is k=1, i=0. Index 1*2+0=2.
    W[2][2] = 1
    
    # p3 = (d-c)*(e+f)
    U[3][3] = 1; U[2][3] = -1
    V[0][3] = 1; V[1][3] = 1
    # C11 is k=1, i=1. Index 1*2+1=3.
    W[3][3] = 1
    
    # p4 = d*h
    U[3][4] = 1; V[3][4] = 1
    W[3][4] = 1
    
    # p5 = (a-b)*(e+h)
    U[0][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[3][5] = 1
    # C00 (k=0,i=0) idx 0, C10 (k=0,i=1) idx 1.
    W[0][5] = 1; W[1][5] = -1
    
    # p6 = (b+c)*(g+e)
    U[1][6] = 1; U[2][6] = 1
    V[2][6] = 1; V[0][6] = 1
    # C10 (k=0,i=1) idx 1, C11 (k=1,i=1) idx 3.
    W[1][6] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    return r, U, V, W

def _mat4x4x4():
    """Rank 49 decomposition for 4x4x4."""
    r = 49
    U = [[0]*r for _ in range(16)]
    V = [[0]*r for _ in range(16)]
    W = [[0]*r for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2()
    
    # Block Strassen Coefficients
    # X_k[u,v] = sum alpha_ab A_ab
    # Y_k[u,v] = sum beta_ab B_ab
    # C_ab = sum gamma_ab,k M_k
    
    # k=0: M0 = (A00+A11)(B00+B11)
    # X0: A00+A11. Y0: B00+B11.
    # C00 += M0, C11 += M0.
    
    # Define alpha, beta, gamma for each k
    # alpha[k][u][v] is a 2x2 matrix of coefficients for blocks A_ab
    # beta[k][u][v] is a 2x2 matrix of coefficients for blocks B_ab
    # gamma[k][a][b] is the coefficient of M_k in C_ab
    
    # We need to map block indices (a,b) to 2x2 matrices of coefficients for the 4x4 entries.
    
    def get_block_coeff(coeff_matrix):
        # coeff_matrix is 2x2, entries are scalars (coeff for A_ab)
        # Returns 4x4 matrix
        res = [[0]*4 for _ in range(4)]
        for a in range(2):
            for b in range(2):
                val = coeff_matrix[a][b]
                if val != 0:
                    for r in range(2):
                        for c in range(2):
                            res[2*a+r][2*b+c] = val
        return res

    # Strassen Block Formulas (0-indexed blocks)
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    block_defs = [
        # k=0
        {
            'X': [[[[1,0],[0,0]], [[0,0],[0,0]]], # u=0,v=0: A00. u=0,v=1: 0
                  [[[0,0],[0,0]], [[0,0],[0,0]]]], # u=1,v=0: 0. u=1,v=1: 0
            # Wait, X0 = A00 + A11.
            # X0[0,0] = A00. X0[0,1] = 0. X0[1,0] = 0. X0[1,1] = A11.
            'X_coeff': [
                [[[1,0],[0,0]], [[0,0],[0,0]]],
                [[[0,0],[0,0]], [[0,0],[1,0]]]
            ],
            'Y_coeff': [
                [[[1,0],[0,0]], [[0,0],[0,0]]],
                [[[0,0],[0,0]], [[0,0],[1,0]]]
            ],
            'Gamma': [
                [1, 0],
                [0, 1]
            ]
        },
        # k=1: M1 = (A10+A11)B00
        {
            'X_coeff': [
                [[[0,0],[0,0]], [[0,0],[0,0]]],
                [[[0,0],[1,0]], [[0,0],[0,0]]] # A10 at (1,0), A11 at (1,1)? No.
                # X1[1,0] = A10+A11? No.
                # (A10+A11) is a 2x2 matrix of blocks? No.
                # A is 2x2 blocks. A10 is bottom-left. A11 is bottom-right.
                # X1 is a 2x2 matrix of blocks.
                # Row 0: 0. Row 1: A10+A11.
                # So X1[1,0] = A10. X1[1,1] = A11.
            ],
            'Y_coeff': [
                [[[1,0],[0,0]], [[0,0],[0,0]]],
                [[[0,0],[0,0]], [[0,0],[0,0]]]
            ],
            'Gamma': [
                [0, 0],
                [1, -1] # C10 += M1, C11 -= M1
            ]
        },
        # k=2: M2 = A00(B01-B11)
        {
            'X_coeff': [
                [[[1,0],[0,0]], [[0,0],[0,0]]],
                [[[0,0],[0,0]], [[0,0],[0,0]]]
            ],
            'Y_coeff': [
                [[[0,0],[1,0]], [[0,0],[0,0]]], # B01 at (0,1). B11 at (1,1) with -1.
                [[[0,0],[0,0]], [[0,0],[-1,0]]]
            ],
            'Gamma': [
                [0, 1],
                [0, 1] # C01 += M2, C11 += M2
            ]
        },
        # k=3: M3 = A11(B10-B00)
        {
            'X_coeff': [
                [[[0,0],[0,0]], [[0,0],[0,0]]],
                [[[0,0],[0,0]], [[0,0],[1,0]]]
            ],
            'Y_coeff': [
                [[[0,0],[0,0]], [[0,0],[0,0]]],
                [[[0,0],[-1,0]], [[0,0],[0,0]]] # B10 at (1,0) with 1. B00 at (0,0) with -1.
            ],
            'Gamma': [
                [1, 0],
                [1, 0] # C00 += M3, C10 += M3
            ]
        },
        # k=4: M4 = (A00+A01)B11
        {
            'X_coeff': [
                [[[1,0],[0,0]], [[0,0],[1,0]]], # A00 at (0,0), A01 at (0,1).
                [[[0,0],[0,0]], [[0,0],[0,0]]]
            ],
            'Y_coeff': [
                [[[0,0],[0,0]], [[0,0],[0,0]]],
                [[[0,0],[0,0]], [[0,0],[1,0]]]
            ],
            'Gamma': [
                [-1, 1], # C00 -= M4, C01 += M4
                [0, 0]
            ]
        },
        # k=5: M5 = (A10-A00)(B00+B01)
        {
            'X_coeff': [
                [[[1,0],[0,0]], [[0,0],[0,0]]], # -A00 at (0,0).
                [[[-1,0],[0,0]], [[0,0],[0,0]]] # A10 at (1,0).
            ],
            'Y_coeff': [
                [[[1,0],[0,0]], [[0,0],[1,0]]], # B00 at (0,0), B01 at (0,1).
                [[[0,0],[0,0]], [[0,0],[0,0]]]
            ],
            'Gamma': [
                [0, 0],
                [0, 1] # C11 += M5
            ]
        },
        # k=6: M6 = (A01-A11)(B10+B11)
        {
            'X_coeff': [
                [[[0,0],[1,0]], [[0,0],[0,0]]], # A01 at (0,1).
                [[[0,0],[0,0]], [[0,0],[-1,0]]] # -A11 at (1,1).
            ],
            'Y_coeff': [
                [[[0,0],[0,0]], [[0,0],[0,0]]],
                [[[0,0],[1,0]], [[0,0],[1,0]]] # B10 at (1,0), B11 at (1,1).
            ],
            'Gamma': [
                [1, 0], # C00 += M6
                [0, 0]
            ]
        }
    ]
    
    # Fix X_coeff for k=1
    block_defs[1]['X_coeff'] = [
        [[[0,0],[0,0]], [[0,0],[0,0]]],
        [[[0,0],[1,0]], [[0,0],[1,0]]] # A10 at (1,0), A11 at (1,1)
    ]
    
    idx = 0
    for k in range(7):
        Xc = block_defs[k]['X_coeff']
        Yc = block_defs[k]['Y_coeff']
        Gamma = block_defs[k]['Gamma']
        
        for c in range(7):
            # u_c, v_c, w_c from Strassen 2x2
            for u in range(2):
                for v in range(2):
                    uc_val = U2[u*2+v][c]
                    vc_val = V2[u*2+v][c]
                    wc_val = W2[u*2+v][c]
                    
                    if uc_val != 0:
                        # Update U
                        coeff_A = get_block_coeff(Xc[u][v])
                        for i in range(4):
                            for j in range(4):
                                val = coeff_A[i][j]
                                if val != 0:
                                    U[i*4+j][idx] += uc_val * val
                                    
                    if vc_val != 0:
                        # Update V
                        coeff_B = get_block_coeff(Yc[u][v])
                        for j in range(4):
                            for l in range(4):
                                val = coeff_B[j][l]
                                if val != 0:
                                    V[j*4+l][idx] += vc_val * val
                                    
                    if wc_val != 0:
                        # Update W
                        for a in range(2):
                            for b in range(2):
                                gamma_val = Gamma[a][b]
                                if gamma_val != 0:
                                    i = 2*a + u
                                    k_out = 2*b + v
                                    W[k_out*4 + i][idx] += wc_val * gamma_val
            idx += 1
    return r, U, V, W

def construct():
    if _check_time():
        n, m, p = 4, 5, 6
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    r1, U1, V1, W1 = _mat4x4x4()
    r2, U2, V2, W2 = _schoolbook(4, 1, 4)
    r3, U3, V3, W3 = _schoolbook(4, 4, 2) # 32
    r4, U4, V4, W4 = _schoolbook(4, 1, 2) # 8
    
    # Optimization: Use Strassen for 4x4x2?
    # r(4,4,2) = 28.
    # Current r3 is 32.
    # Let's implement 4x4x2 with rank 28.
    # 4x4x2 can be seen as two 4x2x2? No.
    # 4x4x2 = (2x2x2) + (2x2x2) + (2x2x2) + (2x2x2)?
    # Blocks: A is 2x2 blocks of 2x2. B is 2x2 blocks of 2x1.
    # This is complex.
    # Let's stick to schoolbook for 4x4x2 for now to ensure correctness, rank 32.
    # Total: 49 + 16 + 32 + 8 = 105.
    # Previous was 116.
    # If I can do 28 for 4x4x2, total is 101.
    
    # Let's try to implement 4x4x2 rank 28.
    # 4x4x2 = 2 * (2x4x2).
    # 2x4x2 rank 14.
    # 2x4x2 = 2 * (2x2x2). Rank 14.
    
    def _mat2x4x2():
        r = 14
        U = [[0]*r for _ in range(8)]
        V = [[0]*r for _ in range(8)]
        W = [[0]*r for _ in range(4)]
        
        U2, V2, W2 = _strassen_2x2()
        
        # Block 0: A cols 0,1. B rows 0,1.
        # A is 2x2 (rows 0,1; cols 0,1). B is 2x2 (rows 0,1; cols 0,1).
        # C is 2x2.
        
        for c in range(7):
            for a in range(4):
                i = a // 2
                j = a % 2
                U[i*4 + j][c] = U2[a][c]
            for b in range(4):
                jj = b // 2
                kk = b % 2
                V[jj*2 + kk][c] = V2[b][c]
            for x in range(4):
                kk = x // 2
                ii = x % 2
                W[kk*2 + ii][c] = W2[x][c]
                
            # Block 1: A cols 2,3. B rows 2,3.
            c1 = 7 + c
            for a in range(4):
                i = a // 2
                j = a % 2
                U[i*4 + (j+2)][c1] = U2[a][c]
            for b in range(4):
                jj = b // 2
                kk = b % 2
                V[(jj+2)*2 + kk][c1] = V2[b][c]
            for x in range(4):
                kk = x // 2
                ii = x % 2
                W[kk*2 + ii][c1] = W2[x][c]
                
        return r, U, V, W

    r3, U3, V3, W3 = _mat2x4x2() # This is 2x4x2. We need 4x4x2.
    # 4x4x2 = 2 * (2x4x2).
    # Top half: A rows 0,1. B rows 0..3. C rows 0,1.
    # Bottom half: A rows 2,3. B rows 0..3. C rows 2,3.
    
    r3_top, U3t, V3t, W3t = _mat2x4x2()
    r3_bot, U3b, V3b, W3b = _mat2x4x2()
    
    r3 = r3_top + r3_bot
    U3 = [[0]*r3 for _ in range(16)]
    V3 = [[0]*r3 for _ in range(8)]
    W3 = [[0]*r3 for _ in range(8)]
    
    for c in range(r3_top):
        for a in range(8):
            i = a // 4
            j = a % 4
            U3[i*4 + j][c] = U3t[a][c]
        for b in range(8):
            j = b // 2
            k = b % 2
            V3[j*2 + k][c] = V3t[b][c]
        for x in range(4):
            k = x // 2
            i = x % 2
            W3[k*4 + i][c] = W3t[x][c]
            
    for c in range(r3_bot):
        c1 = r3_top + c
        for a in range(8):
            i = a // 4
            j = a % 4
            U3[(i+2)*4 + j][c1] = U3b[a][c]
        for b in range(8):
            j = b // 2
            k = b % 2
            V3[j*2 + k][c1] = V3b[b][c]
        for x in range(4):
            k = x // 2
            i = x % 2
            W3[(k+2)*4 + i][c1] = W3b[x][c]
            
    total = r1 + r2 + r3 + r4
    
    n, m, p = 4, 5, 6
    U_g = [[0]*total for _ in range(n*m)]
    V_g = [[0]*total for _ in range(m*p)]
    W_g = [[0]*total for _ in range(p*n)]
    
    base = 0
    # Term 1: A1 (4x4) x B1 (4x4)
    for c in range(r1):
        for a in range(16):
            i = a // 4
            j = a % 4
            U_g[i*5 + j][base + c] = U1[a][c]
        for b in range(16):
            j = b // 4
            k = b % 4
            V_g[j*6 + k][base + c] = V1[b][c]
        for x in range(16):
            k = x // 4
            i = x % 4
            W_g[k*4 + i][base + c] = W1[x][c]
    base += r1
    
    # Term 2: A2 (4x1) x B3 (1x4)
    for c in range(r2):
        for a in range(4):
            i = a
            j = 4
            U_g[i*5 + j][base + c] = U2[a][c]
        for b in range(4):
            j = 4
            k = b
            V_g[j*6 + k][base + c] = V2[b][c]
        for x in range(16):
            k = x // 4
            i = x % 4
            W_g[k*4 + i][base + c] = W2[x][c]
    base += r2
    
    # Term 3: A1 (4x4) x B2 (4x2)
    for c in range(r3):
        for a in range(16):
            i = a // 4
            j = a % 4
            U_g[i*5 + j][base + c] = U3[a][c]
        for b in range(8):
            j = b // 2
            k = b % 2
            V_g[j*6 + (k+4)][base + c] = V3[b][c]
        for x in range(8):
            k = x // 4
            i = x % 4
            W_g[(k+4)*4 + i][base + c] = W3[x][c]
    base += r3
    
    # Term 4: A2 (4x1) x B4 (1x2)
    for c in range(r4):
        for a in range(4):
            i = a
            j = 4
            U_g[i*5 + j][base + c] = U4[a][c]
        for b in range(2):
            j = 4
            k = b
            V_g[j*6 + (k+4)][base + c] = V4[b][c]
        for x in range(8):
            k = x // 4
            i = x % 4
            W_g[(k+4)*4 + i][base + c] = W4[x][c]
    base += r4
    
    return {"U": U_g, "V": V_g, "W": W_g}

