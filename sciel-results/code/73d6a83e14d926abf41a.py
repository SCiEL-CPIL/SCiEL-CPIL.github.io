import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x14 matrix multiplication.
    
    Uses block-wise Strassen: 7 independent 2x2 blocks, each with rank 7.
    Total rank = 49.
    
    U: (4, 49) - coefficients for A (2x2, row-major)
    V: (28, 49) - coefficients for B (2x14, row-major)
    W: (28, 49) - coefficients for C (2x14, column-major: W[k*n+i, r])
    """
    start = time.time()
    deadline = 145.0  # safety margin before 150s kill

    # Strassen for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6

    A_c = [
        [1,0,0,1],   # M1: a+d
        [0,0,1,1],   # M2: c+d
        [1,0,0,0],   # M3: a
        [0,0,0,1],   # M4: d
        [1,1,0,0],   # M5: a+b
        [-1,0,1,0],  # M6: c-a
        [0,1,0,-1],  # M7: b-d
    ]
    B_c = [
        [1,0,0,1],   # M1: e+h
        [1,0,0,0],   # M2: e
        [0,1,0,-1],  # M3: f-h
        [-1,0,1,0],  # M4: g-e
        [0,0,0,1],   # M5: h
        [1,1,0,0],   # M6: e+f
        [0,0,1,1],   # M7: g+h
    ]
    C_c = [
        [1,0,0,1],    # M1: C00+1, C11+1
        [0,0,1,-1],   # M2: C10+1, C11-1
        [0,1,0,1],    # M3: C01+1, C11+1
        [1,0,1,0],    # M4: C00+1, C10+1
        [-1,1,0,0],   # M5: C00-1, C01+1
        [0,0,0,1],    # M6: C11+1
        [1,0,0,0],    # M7: C00+1
    ]

    n, m, p = 2, 2, 14
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for b in range(num_blocks):
        for s in range(7):
            col = b*7 + s
            # U: A is 2x2 row-major [a,b,c,d]
            for i in range(4):
                U[i][col] = A_c[s][i]
            # V: B is 2x14 row-major, block b uses cols 2b,2b+1
            # e=B[0,2b]=idx 2b, f=B[0,2b+1]=idx 2b+1
            # g=B[1,2b]=idx 14+2b, h=B[1,2b+1]=idx 15+2b
            ie, if_, ig, ih = 2*b, 2*b+1, 14+2*b, 15+2*b
            V[ie][col] = B_c[s][0]
            V[if_][col] = B_c[s][1]
            V[ig][col] = B_c[s][2]
            V[ih][col] = B_c[s][3]
            # W: C is 2x14, column-major W[k*n+i] where k=col, i=row
            # C[0,2b] -> idx 2b*2+0=4b, C[1,2b] -> idx 2b*2+1=4b+1
            # C[0,2b+1] -> idx (2b+1)*2+0=4b+2, C[1,2b+1] -> idx (2b+1)*2+1=4b+3
            W[4*b][col] = C_c[s][0]       # C00
            W[4*b+1][col] = C_c[s][2]     # C10
            W[4*b+2][col] = C_c[s][1]     # C01
            W[4*b+3][col] = C_c[s][3]     # C11

    return {"U": U, "V": V, "W": W}
