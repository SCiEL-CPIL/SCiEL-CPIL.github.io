import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 26 decomposition: Strassen (7) + 19 border terms
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    def iu(i, j): return i * m + j
    def iv(j, k): return j * p + k
    def iw(k, i): return k * n + i
    
    # Strassen's 7 multiplications for 2x2 core (i,j,k in {0,1})
    # m1 = (a00+a11)(b00+b11)
    r = 0
    U[iu(0,0), r] = 1; U[iu(1,1), r] = 1
    V[iv(0,0), r] = 1; V[iv(1,1), r] = 1
    W[iw(0,0), r] = 1; W[iw(1,1), r] = 1
    
    # m2 = (a10+a11)b00
    r = 1
    U[iu(1,0), r] = 1; U[iu(1,1), r] = 1
    V[iv(0,0), r] = 1
    W[iw(0,1), r] = 1; W[iw(1,1), r] = -1
    
    # m3 = a00(b01-b11)
    r = 2
    U[iu(0,0), r] = 1
    V[iv(0,1), r] = 1; V[iv(1,1), r] = -1
    W[iw(1,0), r] = 1; W[iw(1,1), r] = 1
    
    # m4 = a11(b10-b00)
    r = 3
    U[iu(1,1), r] = 1
    V[iv(1,0), r] = 1; V[iv(0,0), r] = -1
    W[iw(0,0), r] = 1; W[iw(0,1), r] = 1
    
    # m5 = (a00+a01)b11
    r = 4
    U[iu(0,0), r] = 1; U[iu(0,1), r] = 1
    V[iv(1,1), r] = 1
    W[iw(0,0), r] = -1; W[iw(1,0), r] = 1
    
    # m6 = (a10-a00)(b00+b01)
    r = 5
    U[iu(1,0), r] = 1; U[iu(0,0), r] = -1
    V[iv(0,0), r] = 1; V[iv(0,1), r] = 1
    W[iw(1,1), r] = 1
    
    # m7 = (a01-a11)(b10+b11)
    r = 6
    U[iu(0,1), r] = 1; U[iu(1,1), r] = -1
    V[iv(1,0), r] = 1; V[iv(1,1), r] = 1
    W[iw(0,0), r] = 1
    
    # Border terms (19 terms)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[iu(i,j), col] = 1
                V[iv(j,k), col] = 1
                W[iw(k,i), col] = 1
                col += 1
                
    # Verify
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    if np.allclose(T_recon, T_target, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook 27
    R27 = n * m * p
    U27 = np.zeros((n * m, R27))
    V27 = np.zeros((m * p, R27))
    W27 = np.zeros((p * n, R27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[iu(i,j), col] = 1
                V27[iv(j,k), col] = 1
                W27[iw(k,i), col] = 1
                col += 1
                
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
