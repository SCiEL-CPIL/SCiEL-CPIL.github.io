import numpy as np
from functools import lru_cache

DIMENSIONS = [3, 5, 6]

def _strassen_2x2_2x2():
    # Returns numpy arrays
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1
    
    # M3
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    
    # M4
    U[3,3]=1
    V[1,3]=1; V[2,3]=-1
    W[0,3]=1; W[1,3]=1
    
    # M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[3,4]=1
    
    # M6
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    W[3,5]=1
    
    # M7
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

@lru_cache(maxsize=None)
def _solve(n, m, p):
    if n == 0 or m == 0 or p == 0:
        return np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0)), 0
        
    if n == 1 and m == 1 and p == 1:
        U = np.ones((1, 1), dtype=int)
        V = np.ones((1, 1), dtype=int)
        W = np.ones((1, 1), dtype=int)
        return U, V, W, 1
        
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2_2x2()
        return U, V, W, 7
        
    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1, r1 = _solve(n1, m, p)
        U2, V2, W2, r2 = _solve(n2, m, p)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            # Assemble
            U = np.zeros((n*m, r), dtype=int)
            V = np.zeros((m*p, r), dtype=int)
            W = np.zeros((p*n, r), dtype=int)
            
            # Place U1
            # U1 rows: i1*m + j, i1 in [0, n1)
            # U rows: (n1_offset + i1)*m + j. Offset 0.
            for i1 in range(n1):
                for j in range(m):
                    U[i1*m+j, :r1] = U1[i1*m+j, :]
            
            # Place U2
            # U2 rows: i2*m + j, i2 in [0, n2)
            # U rows: (n1 + i2)*m + j
            for i2 in range(n2):
                for j in range(m):
                    U[(n1+i2)*m+j, r1:] = U2[i2*m+j, :]
            
            # Place V1, V2 (hstack)
            V[:, :r1] = V1
            V[:, r1:] = V2
            
            # Place W1, W2
            # W1 rows: k*n1 + i1
            # W rows: k*n + i1 (i1 < n1)
            for k in range(p):
                for i1 in range(n1):
                    W[k*n+i1, :r1] = W1[k*n1+i1, :]
            
            # W2 rows: k*n2 + i2
            # W rows: k*n + (n1 + i2)
            for k in range(p):
                for i2 in range(n2):
                    W[k*n + (n1+i2), r1:] = W2[k*n2+i2, :]
            
            best_U = U
            best_V = V
            best_W = W
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1, r1 = _solve(n, m1, p)
        U2, V2, W2, r2 = _solve(n, m2, p)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            U = np.zeros((n*m, r), dtype=int)
            V = np.zeros((m*p, r), dtype=int)
            W = np.zeros((p*n, r), dtype=int)
            
            # Place U1, U2 (hstack)
            U[:, :r1] = U1
            U[:, r1:] = U2
            
            # Place V1
            # V1 rows: j1*p + k, j1 in [0, m1)
            # V rows: j1*p + k
            for j1 in range(m1):
                for k in range(p):
                    V[j1*p+k, :r1] = V1[j1*p+k, :]
            
            # Place V2
            # V2 rows: j2*p + k, j2 in [0, m2)
            # V rows: (m1+j2)*p + k
            for j2 in range(m2):
                for k in range(p):
                    V[(m1+j2)*p+k, r1:] = V2[j2*p+k, :]
            
            # Place W1, W2 (hstack)
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_U = U
            best_V = V
            best_W = W

    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1, r1 = _solve(n, m, p1)
        U2, V2, W2, r2 = _solve(n, m, p2)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            U = np.zeros((n*m, r), dtype=int)
            V = np.zeros((m*p, r), dtype=int)
            W = np.zeros((p*n, r), dtype=int)
            
            # Place U1, U2 (hstack)
            U[:, :r1] = U1
            U[:, r1:] = U2
            
            # Place V1
            # V1 rows: j*p1 + k1, k1 in [0, p1)
            # V rows: j*p1 + k1
            for j in range(m):
                for k1 in range(p1):
                    V[j*p+k1, :r1] = V1[j*p1+k1, :]
            
            # Place V2
            # V2 rows: j*p2 + k2, k2 in [0, p2)
            # V rows: j*p + (p1 + k2)
            for j in range(m):
                for k2 in range(p2):
                    V[j*p + (p1+k2), r1:] = V2[j*p2+k2, :]
            
            # Place W1, W2 (vstack)
            W[:p1*n, :r1] = W1
            W[p1*n:, r1:] = W2
            
            best_U = U
            best_V = V
            best_W = W

    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _solve(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
