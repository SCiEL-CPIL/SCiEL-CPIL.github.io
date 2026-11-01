"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """Return a rank-49 exact tensor decomposition for 4x4 matrix multiplication."""
    import numpy as np
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Correct Strassen 2x2 factors
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-c),
    # M5 = (a+b)h, M6 = (c-d)(g+h), M7 = (a-b)(e+f)
    # C[0,0] = M1+M4-M5+M7, C[0,1] = M3+M5, C[1,0] = M2+M4, C[1,1] = M1-M2+M3+M6
    
    # U2[i*2+j, r] for i=row of A, j=col of A
    # A = [[a,b],[c,d]], so:
    # row 0: a (i=0,j=0), b (i=0,j=1)
    # row 1: c (i=1,j=0), d (i=1,j=1)
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 1],   # a: M1,M3,M5,M7
        [0, 0, 0, 0, 1, 0,-1],   # b: M5,M7
        [0, 1, 0, 0, 0, 1, 0],   # c: M2,M6
        [1, 1, 0, 1, 0,-1, 0],   # d: M1,M2,M4,M6
    ], dtype=float)
    
    # V2[j*2+k, r] for j=row of B, k=col of B
    # B = [[e,f],[g,h]], so:
    # row 0: e (j=0,k=0), f (j=0,k=1)
    # row 1: g (j=1,k=0), h (j=1,k=1)
    V2 = np.array([
        [1, 1, 0,-1, 0, 0, 1],   # e: M1,M2,M7
        [0, 0, 1, 0, 0, 0, 1],   # f: M3,M7
        [0, 0, 0, 1, 0, 1, 0],   # g: M4,M6
        [1, 0, 0, 0, 1, 1, 0],   # h: M1,M5,M6
    ], dtype=float)
    
    # W2[k*2+i, r] for k=col of C, i=row of C
    # C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6
    W2 = np.array([
        [1, 0, 0, 1,-1, 0, 1],   # C[0,0] (k=0,i=0)
        [0, 1, 0, 0, 0, 0, 0],   # C[1,0] (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1] (k=1,i=0)
        [1,-1, 1, 0, 0, 1, 0],   # C[1,1] (k=1,i=1)
    ], dtype=float)
    
    # Verify 2x2 decomposition
    valid_2x2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j] * V2[j*2+k] * W2[k*2+i])
                if abs(val - 1.0) > 1e-10:
                    valid_2x2 = False
                    break
            if not valid_2x2:
                break
        if not valid_2x2:
            break
    
    if valid_2x2:
        # Build 4x4 via Kronecker product: rank = 7*7 = 49
        rank = 49
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for i in range(n):
            for j in range(m):
                i1, i2 = i // 2, i % 2
                j1, j2 = j // 2, j % 2
                for r1 in range(7):
                    for r2 in range(7):
                        r = r1 * 7 + r2
                        U[i*m + j, r] = U2[i1*2 + i2, r1] * U2[j1*2 + j2, r2]
        
        for j in range(m):
            for k in range(p):
                j1, j2 = j // 2, j % 2
                k1, k2 = k // 2, k % 2
                for r1 in range(7):
                    for r2 in range(7):
                        r = r1 * 7 + r2
                        V[j*p + k, r] = V2[j1*2 + j2, r1] * V2[k1*2 + k2, r2]
        
        for k in range(p):
            for i in range(n):
                k1, k2 = k // 2, k % 2
                i1, i2 = i // 2, i % 2
                for r1 in range(7):
                    for r2 in range(7):
                        r = r1 * 7 + r2
                        W[k*n + i, r] = W2[k1*2 + k2, r1] * W2[i1*2 + i2, r2]
        
        # Verify 4x4 decomposition with vectorized check
        if time.time() < deadline - 10:
            # Vectorized verification: for each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
            valid_4x4 = True
            for i in range(n):
                if time.time() > deadline - 10:
                    valid_4x4 = False
                    break
                for j in range(m):
                    if time.time() > deadline - 10:
                        valid_4x4 = False
                        break
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        if abs(val - 1.0) > 1e-10:
                            valid_4x4 = False
                            break
                    if not valid_4x4:
                        break
                if not valid_4x4:
                    break
            
            if valid_4x4:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 64
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
