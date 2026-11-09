import time
import numpy as np

DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _build_rank_40():
    n, m, p = 2, 4, 5
    # Rank 40 decomposition using schoolbook for rectangular blocks
    # Part 1: A[0:2, 0:2] * B[0:2, 0:4] -> 16 terms
    # Part 2: A[0:2, 2:4] * B[2:4, 0:4] -> 16 terms
    # Part 3: A[0:2, 0:4] * B[0:4, 4]   -> 8 terms
    
    rank = 40
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Part 1: Rows 0-1 of A, Cols 0-1 of B (which are Rows 0-1 of B)
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Part 2: Rows 0-1 of A, Cols 2-3 of A (which are Rows 2-3 of B)
    for i in range(2):
        for j in range(2, 4):
            for k in range(4):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Part 3: Rows 0-1 of A, Cols 0-3 of A, Col 4 of B
    for i in range(2):
        for j in range(4):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    return U, V, W

def _verify(U, V, W):
    n, m, p = 2, 4, 5
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] += u[i,j]*v[j,k]*w[k,i]
                    
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Texp[i*m+j, j*p+k, k*n+i] = 1.0
                
    return np.allclose(T, Texp, atol=1e-9)

def construct():
    n, m, p = 2, 4, 5
    # Use the Strassen-hybrid decomposition which achieves rank 36
    # This is superior to the schoolbook rank 40 decomposition
    U, V, W = _build_seed_36()
    
    # Verify correctness before returning
    if not _verify(U, V, W):
        # Fallback to schoolbook rank 40 if Strassen-hybrid fails (should not happen)
        U, V, W = _build_rank_40()
        
    return {"U": U, "V": V, "W": W}

def _build_seed_36():
    n, m, p = 2, 4, 5
    strassen_U = [
        [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
        [1,1,0,0],[-1,0,1,0],[0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
        [0,0,0,1],[1,1,0,0],[0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
        [-1,0,1,0],[0,0,0,1],[1,0,0,0],
    ]
    rank = 4*7 + 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for j0 in (0,2):
        for k0 in (0,2):
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                col += 1
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    return U, V, W
