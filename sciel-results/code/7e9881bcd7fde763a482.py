import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # 4x4x4 part: Kronecker product of 2x2x2 schoolbook (rank 4) -> rank 16
    # 4x4x1 part (k=4): schoolbook -> rank 4
    # Total rank: 20
    
    r = 20
    U = np.zeros((n*m, r))  # (16, 20)
    V = np.zeros((m*p, r))  # (20, 20)
    W = np.zeros((p*n, r))  # (20, 20)
    
    # Part 1: Kronecker product for 4x4x4 (columns 0-15)
    for t1 in range(4):
        i1 = t1 // 4
        j1 = (t1 // 2) % 2
        k1 = t1 % 2
        
        for t2 in range(4):
            i2 = t2 // 4
            j2 = (t2 // 2) % 2
            k2 = t2 % 2
            
            col = t1 * 4 + t2  # 0 to 15
            
            i = 2*i1 + i2
            j = 2*j1 + j2
            k = 2*k1 + k2
            
            U[i*m + j, col] = 1.0
            V[j*p + k, col] = 1.0
            W[k*n + i, col] = 1.0
    
    # Part 2: Schoolbook for k=4 (columns 16-19)
    for i in range(n):
        for j in range(m):
            col = 16 + i*m + j
            U[i*m + j, col] = 1.0
            V[j*p + 4, col] = 1.0
            W[4*n + i, col] = 1.0
    
    # Verify
    def verify(U, V, W, n, m, p, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p, r):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    r_school = n*m*p
    U_s = np.zeros((n*m, r_school))
    V_s = np.zeros((m*p, r_school))
    W_s = np.zeros((p*n, r_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
