import numpy as np
import time

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
DEADLINE = 145  # safety margin before 150s kill

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> C[:, 0:6]
    # A[i, 0] * B[0, k] -> C[i, k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> C[:, 0:6]
    # Split B[1:3, :] into three 2x2 blocks: cols (0,1), (2,3), (4,5)
    # Each block uses Strassen with 7 multiplications
    
    for t in range(3):
        c0 = 2 * t
        c1 = 2 * t + 1
        
        # Strassen M1: (A00+A11)(B00+B11)
        U[0*m + 1, col] += 1.0
        U[1*m + 2, col] += 1.0
        V[1*p + c0, col] += 1.0
        V[2*p + c1, col] += 1.0
        W[c0*n + 0, col] += 1.0
        W[c1*n + 1, col] += 1.0
        col += 1
        
        # Strassen M2: (A10+A11)B00
        U[1*m + 1, col] += 1.0
        U[1*m + 2, col] += 1.0
        V[1*p + c0, col] += 1.0
        W[c0*n + 1, col] += 1.0
        W[c1*n + 1, col] += -1.0
        col += 1
        
        # Strassen M3: A00(B01-B11)
        U[0*m + 1, col] += 1.0
        V[1*p + c1, col] += 1.0
        V[2*p + c1, col] += -1.0
        W[c1*n + 0, col] += 1.0
        W[c1*n + 1, col] += 1.0
        col += 1
        
        # Strassen M4: A11(B10-B00)
        U[1*m + 2, col] += 1.0
        V[2*p + c0, col] += 1.0
        V[1*p + c0, col] += -1.0
        W[c0*n + 0, col] += 1.0
        W[c0*n + 1, col] += 1.0
        col += 1
        
        # Strassen M5: (A00+A01)B11
        U[0*m + 1, col] += 1.0
        U[0*m + 2, col] += 1.0
        V[2*p + c1, col] += 1.0
        W[c0*n + 0, col] += -1.0
        W[c1*n + 0, col] += 1.0
        col += 1
        
        # Strassen M6: (A10-A00)(B00+B01)
        U[1*m + 1, col] += 1.0
        U[0*m + 1, col] += -1.0
        V[1*p + c0, col] += 1.0
        V[1*p + c1, col] += 1.0
        W[c1*n + 1, col] += 1.0
        col += 1
        
        # Strassen M7: (A01-A11)(B10+B11)
        U[0*m + 2, col] += 1.0
        U[1*m + 2, col] += -1.0
        V[2*p + c0, col] += 1.0
        V[2*p + c1, col] += 1.0
        W[c0*n + 0, col] += 1.0
        col += 1

    # Verify correctness via tensor contraction
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj, c] = 1.0
                    V_fb[jj * p + kk, c] = 1.0
                    W_fb[kk * n + ii, c] = 1.0
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
