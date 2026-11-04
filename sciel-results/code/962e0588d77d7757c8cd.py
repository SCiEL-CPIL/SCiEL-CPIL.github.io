import numpy as np
import time

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if i == i2 and j == j2 and k == k2:
                    continue
                actual = 0.0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if abs(actual) > tol:
                    return False
    return True

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 115.0

    # Try rank-23 Winograd decomposition
    U23 = np.zeros((9, 23), dtype=float)
    V23 = np.zeros((9, 23), dtype=float)
    W23 = np.zeros((9, 23), dtype=float)

    # Direct terms (0-8)
    U23[0,0]=1; V23[0,0]=1; W23[0,0]=1
    U23[1,1]=1; V23[3,1]=1; W23[0,1]=1
    U23[2,2]=1; V23[6,2]=1; W23[0,2]=1
    U23[3,3]=1; V23[1,3]=1; W23[3,3]=1
    U23[4,4]=1; V23[4,4]=1; W23[4,4]=1
    U23[5,5]=1; V23[7,5]=1; W23[4,5]=1
    U23[6,6]=1; V23[2,6]=1; W23[6,6]=1
    U23[7,7]=1; V23[5,7]=1; W23[7,7]=1
    U23[8,8]=1; V23[8,8]=1; W23[8,8]=1

    # Winograd terms (9-22)
    U23[0,9]=1; U23[1,9]=1
    V23[1,9]=1; V23[4,9]=1
    W23[3,9]=1; W23[4,9]=1; W23[1,9]=-1; W23[5,9]=-1

    U23[0,10]=1; U23[2,10]=1
    V23[2,10]=1; V23[8,10]=1
    W23[6,10]=1; W23[8,10]=1; W23[2,10]=-1; W23[7,10]=-1

    U23[1,11]=1; U23[2,11]=1
    V23[5,11]=1; V23[7,11]=1
    W23[7,11]=1; W23[8,11]=1; W23[5,11]=-1; W23[6,11]=-1

    U23[3,12]=1; U23[4,12]=1
    V23[1,12]=1; V23[4,12]=1
    W23[3,12]=1; W23[4,12]=1; W23[1,12]=-1; W23[5,12]=-1

    U23[3,13]=1; U23[5,13]=1
    V23[2,13]=1; V23[8,13]=1
    W23[6,13]=1; W23[8,13]=1; W23[2,13]=-1; W23[7,13]=-1

    U23[4,14]=1; U23[5,14]=1
    V23[5,14]=1; V23[7,14]=1
    W23[7,14]=1; W23[8,14]=1; W23[5,14]=-1; W23[6,14]=-1

    U23[6,15]=1; U23[7,15]=1
    V23[1,15]=1; V23[4,15]=1
    W23[3,15]=1; W23[4,15]=1; W23[1,15]=-1; W23[5,15]=-1

    U23[6,16]=1; U23[8,16]=1
    V23[2,16]=1; V23[8,16]=1
    W23[6,16]=1; W23[8,16]=1; W23[2,16]=-1; W23[7,16]=-1

    U23[7,17]=1; U23[8,17]=1
    V23[5,17]=1; V23[7,17]=1
    W23[7,17]=1; W23[8,17]=1; W23[5,17]=-1; W23[6,17]=-1

    U23[0,18]=1; U23[3,18]=1
    V23[0,18]=1; V23[3,18]=1
    W23[0,18]=1; W23[3,18]=1; W23[1,18]=-1; W23[6,18]=-1

    U23[0,19]=1; U23[6,19]=1
    V23[2,19]=1; V23[3,19]=1
    W23[0,19]=1; W23[6,19]=1; W23[1,19]=-1; W23[3,19]=-1

    U23[3,20]=1; U23[6,20]=1
    V23[0,20]=1; V23[3,20]=1
    W23[3,20]=1; W23[6,20]=1; W23[1,20]=-1; W23[2,20]=-1

    U23[1,21]=1; U23[4,21]=1
    V23[3,21]=1; V23[4,21]=1
    W23[3,21]=1; W23[4,21]=1; W23[5,21]=-1; W23[7,21]=-1

    U23[1,22]=1; U23[4,22]=1
    V23[5,22]=1; V23[8,22]=1
    W23[6,22]=1; W23[8,22]=1; W23[5,22]=-1; W23[7,22]=-1

    # Verify the rank-23 decomposition
    if _verify_decomposition(U23, V23, W23):
        return {"U": U23.tolist(), "V": V23.tolist(), "W": W23.tolist()}

    # Fallback: schoolbook rank-27
    rank = 27
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((9, rank), dtype=int)
    W = np.zeros((9, rank), dtype=int)

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
