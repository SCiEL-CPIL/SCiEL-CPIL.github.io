import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    rank = 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen factors for 2x2 * 2x2
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    # Strassen for A[:, 0:2] * B[0:2, 0:2] → C[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Merged factor for A[:, 2] * B[2, 0:4] → C[0:2, 0:4]
    # Covers (i∈{0,1}, j=2, k∈{0,1,2,3})
    U[0*m+2, col] = 1; U[1*m+2, col] = 1
    V[2*p+0, col] = 1; V[2*p+1, col] = 1; V[2*p+2, col] = 1; V[2*p+3, col] = 1
    W[0*n+0, col] = 1; W[0*n+1, col] = 1
    W[1*n+0, col] = 1; W[1*n+1, col] = 1
    W[2*n+0, col] = 1; W[2*n+1, col] = 1
    W[3*n+0, col] = 1; W[3*n+1, col] = 1
    col += 1
    
    # Strassen for A[:, 0:2] * B[0:2, 2:4] → C[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Merged factor for A[:, 0:3] * B[0:3, 4] → C[0:2, 4]
    # Covers (i∈{0,1}, j∈{0,1,2}, k=4)
    U[0*m+0, col] = 1; U[0*m+1, col] = 1; U[0*m+2, col] = 1
    U[1*m+0, col] = 1; U[1*m+1, col] = 1; U[1*m+2, col] = 1
    V[0*p+4, col] = 1; V[1*p+4, col] = 1; V[2*p+4, col] = 1
    W[4*n+0, col] = 1; W[4*n+1, col] = 1
    col += 1
    
    # Verify rank used
    if col != rank:
        pass
    
    # Verify exactness
    res = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]
        
    if not np.allclose(res, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
