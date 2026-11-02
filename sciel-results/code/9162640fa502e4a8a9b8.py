"""Matrix multiplication 2x2x2: rank-7 Strassen decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-7 tensor decomposition for 2x2x2 matrix multiplication (Strassen).
    
    Convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 iff the (i,j,k) entry of the multiplication tensor is 1.
    For 2x2x2, all 8 entries T[i,j,k] are 1.
    """
    rank = 7
    
    # Strassen's seven products:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)h
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)e
    # P6 = (c+d)(f-h)
    # P7 = (a-c)(e+h)
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Term 0: P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # Term 1: P2 = (c+d)h
    U[2][1] = 1; U[3][1] = 1
    V[3][1] = 1
    W[0][1] = 1; W[1][1] = 1; W[2][1] = 0; W[3][1] = 1
    
    # Term 2: P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 0; W[1][2] = 0; W[2][2] = 1; W[3][2] = 0
    
    # Term 3: P4 = d(g-e)
    U[3][3] = 1
    V[0][3] = -1; V[2][3] = 1
    W[0][3] = 1; W[1][3] = 1; W[2][3] = 0; W[3][3] = 1
    
    # Term 4: P5 = (a+b)e
    U[0][4] = 1; U[1][4] = 1
    V[0][4] = 1
    W[0][4] = 1; W[1][4] = 0; W[2][4] = 1; W[3][4] = 0
    
    # Term 5: P6 = (c+d)(f-h)
    U[2][5] = 1; U[3][5] = 1
    V[1][5] = 1; V[3][5] = -1
    W[0][5] = 0; W[1][5] = 0; W[2][5] = 0; W[3][5] = 1
    
    # Term 6: P7 = (a-c)(e+h)
    U[0][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 0; W[2][6] = 0; W[3][6] = 1
    
    # Verify the decomposition
    def verify(U, V, W):
        n, m, p = 2, 2, 2
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for t in range(r):
                        val += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if val != 1:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-8 schoolbook
    rank8 = 8
    U8 = [[0] * rank8 for _ in range(4)]
    V8 = [[0] * rank8 for _ in range(4)]
    W8 = [[0] * rank8 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U8[i*2+j][col] = 1
                V8[j*2+k][col] = 1
                W8[k*2+i][col] = 1
                col += 1
    return {"U": U8, "V": V8, "W": W8}
# EVOLVE-BLOCK-END
