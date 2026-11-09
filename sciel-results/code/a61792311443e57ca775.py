"""Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build Strassen decomposition for 2x2 * 2x6 (rank 7)
    # Strassen products for [[a,b],[c,d]] * [[e,f],[g,h]]:
    # m1=(a+b)(e+f), m2=(a+b)(g+h), m3=d(e-g), m4=a(h-f)
    # m5=(c+d)e, m6=(a-c)(f+h), m7=(b-d)(g+h)
    # C[0,0]=m1+m3-m5-m7, C[0,1]=m4+m5, C[1,0]=m2-m4, C[1,1]=m1-m2+m6+m7
    
    # For 2x2 * 2x6, extend each 2x2 column with the same Strassen pattern
    # U rows: 0=a, 1=b, 2=c, 3=d
    # V rows: 0=e, 1=f, 2=g, 3=h (for each column pair)
    # W rows: 0=C00, 1=C01, 2=C10, 3=C11 (for each column)
    
    # Strassen U factors (4x7):
    # m1: a+b -> [1,1,0,0]
    # m2: a+b -> [1,1,0,0]
    # m3: d   -> [0,0,0,1]
    # m4: a   -> [1,0,0,0]
    # m5: c+d -> [0,0,1,1]
    # m6: a-c -> [1,0,-1,0]
    # m7: b-d -> [0,1,0,-1]
    strassen_U = [
        [1, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, -1, 0],
        [0, 0, 1, 0, 1, 0, -1],
    ]
    
    # Strassen V factors for 2x6 (12x7):
    # For column pair (2c, 2c+1): e=V[2c], f=V[2c+1], g=V[4+2c], h=V[4+2c+1]
    # m1: e+f -> [1,1, 0,0, 0,0, ...] for each pair
    # m2: g+h -> [0,0, 1,1, 0,0, ...]
    # m3: e-g -> [1,-1, 0,0, 0,0, ...]
    # m4: h-f -> [0,0, -1,1, 0,0, ...]
    # m5: e   -> [1,0, 0,0, 0,0, ...]
    # m6: f+h -> [0,1, 0,1, 0,0, ...]
    # m7: g+h -> [0,0, 1,1, 0,0, ...]
    
    strassen_V = [[0]*7 for _ in range(12)]
    for c in range(3):  # 3 column pairs
        base = 2 * c
        # m1: e+f
        strassen_V[base][0] = 1
        strassen_V[base+1][0] = 1
        # m2: g+h
        strassen_V[base+2][1] = 1
        strassen_V[base+3][1] = 1
        # m3: e-g
        strassen_V[base][2] = 1
        strassen_V[base+2][2] = -1
        # m4: h-f
        strassen_V[base+1][3] = -1
        strassen_V[base+3][3] = 1
        # m5: e
        strassen_V[base][4] = 1
        # m6: f+h
        strassen_V[base+1][5] = 1
        strassen_V[base+3][5] = 1
        # m7: g+h
        strassen_V[base+2][6] = 1
        strassen_V[base+3][6] = 1
    
    # Strassen W factors for 2x6 output (12x7):
    # Output row k*n+i: for n=2, rows are 0=C00,1=C01,2=C10,3=C11 for col0
    #                  4=C00,5=C01,6=C10,7=C11 for col1
    #                  8=C00,9=C01,10=C10,11=C11 for col2
    # C00 = m1+m3-m5-m7
    # C01 = m4+m5
    # C10 = m2-m4
    # C11 = m1-m2+m6+m7
    strassen_W = [[0]*7 for _ in range(12)]
    for c in range(3):
        base = 4 * c
        # C00 = m1+m3-m5-m7
        strassen_W[base][0] = 1
        strassen_W[base][2] = 1
        strassen_W[base][4] = -1
        strassen_W[base][6] = -1
        # C01 = m4+m5
        strassen_W[base+1][3] = 1
        strassen_W[base+1][4] = 1
        # C10 = m2-m4
        strassen_W[base+2][1] = 1
        strassen_W[base+2][3] = -1
        # C11 = m1-m2+m6+m7
        strassen_W[base+3][0] = 1
        strassen_W[base+3][1] = -1
        strassen_W[base+3][5] = 1
        strassen_W[base+3][6] = 1
    
    # Schoolbook for 2x3 * 3x6 (rank 18)
    # U rows: 4,5,6 (a[0,2],a[0,3],a[0,4] are rows 2,3,4; a[1,2],a[1,3],a[1,4] are rows 7,8,9)
    # Wait, let me re-index. For the 2x3 part:
    # A rows: row0 = [a02, a03, a04], row1 = [a12, a13, a14]
    # In U indexing: i*m+j where i=0,1 and j=2,3,4
    # So U rows for this part: 0*5+2=2, 0*5+3=3, 0*5+4=4, 1*5+2=7, 1*5+3=8, 1*5+4=9
    # But we're building a separate 2x3*3x6 problem, so let's use local indices
    # Local U rows: 0,1,2 for row0 cols 0,1,2 and 3,4,5 for row1 cols 0,1,2
    # Local V rows: 0,1,2,3,4,5,6,7,8,9,10,11 for 3x6
    # Local W rows: 0,1,2,3,4,5 for 2x6 output
    
    schoolbook_U = [[0]*18 for _ in range(6)]
    schoolbook_V = [[0]*18 for _ in range(18)]
    schoolbook_W = [[0]*18 for _ in range(12)]
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(6):
                schoolbook_U[i*3 + j][col] = 1
                schoolbook_V[j*6 + k][col] = 1
                schoolbook_W[k*2 + i][col] = 1
                col += 1
    
    # Now combine: full U is (10, 25), V is (30, 25), W is (12, 25)
    rank = 25
    U = [[0]*rank for _ in range(10)]
    V = [[0]*rank for _ in range(30)]
    W = [[0]*rank for _ in range(12)]
    
    # Fill Strassen part (columns 0-6)
    # U rows 0-3 (i=0,1; j=0,1) get strassen_U
    for r in range(4):
        for c in range(7):
            U[r][c] = strassen_U[r][c]
    
    # V rows 0-11 (j=0,1; k=0-5) get strassen_V
    for r in range(12):
        for c in range(7):
            V[r][c] = strassen_V[r][c]
    
    # W rows 0-11 get strassen_W
    for r in range(12):
        for c in range(7):
            W[r][c] = strassen_W[r][c]
    
    # Fill schoolbook part (columns 7-24)
    # U rows for j=2,3,4: rows 2,3,4 (i=0) and 7,8,9 (i=1)
    # In local schoolbook_U: row i*3+j_local where j_local = j-2
    for i in range(2):
        for j_local in range(3):
            j = j_local + 2  # global j index
            u_global = i * 5 + j  # global U row
            for k in range(6):
                c = 7 + i * 18 + j_local * 6 + k
                U[u_global][c] = schoolbook_U[i*3 + j_local][i*18 + j_local*6 + k]
                V[j * 6 + k][c] = schoolbook_V[j_local*6 + k][i*18 + j_local*6 + k]
                W[k * 2 + i][c] = schoolbook_W[k*2 + i][i*18 + j_local*6 + k]
    
    # Verify
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i*m + j, c] * Va[j*p + k, c] * Wa[k*n + i, c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
