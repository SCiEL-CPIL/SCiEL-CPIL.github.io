import time
import sys

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    rank = 49  # 7 blocks * 7 Strassen terms
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for one 2x2 block
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        r_base = 7 * b
        
        # Term 0: M1
        r = r_base + 0
        U[0*m + 0][r] += 1
        U[1*m + 1][r] += 1
        V[0*p + k0][r] += 1
        V[1*p + k1][r] += 1
        W[k0*n + 0][r] += 1
        W[k1*n + 1][r] += 1
        
        # Term 1: M2
        r = r_base + 1
        U[1*m + 0][r] += 1
        U[1*m + 1][r] += 1
        V[0*p + k0][r] += 1
        W[k0*n + 1][r] += 1
        W[k1*n + 1][r] += -1
        
        # Term 2: M3
        r = r_base + 2
        U[0*m + 0][r] += 1
        V[1*p + k0][r] += 1
        V[1*p + k1][r] += -1
        W[k1*n + 0][r] += 1
        W[k1*n + 1][r] += 1
        
        # Term 3: M4
        r = r_base + 3
        U[1*m + 1][r] += 1
        V[0*p + k0][r] += 1
        V[1*p + k0][r] += -1
        W[k0*n + 0][r] += 1
        W[k0*n + 1][r] += 1
        
        # Term 4: M5
        r = r_base + 4
        U[0*m + 0][r] += 1
        U[0*m + 1][r] += 1
        V[1*p + k1][r] += 1
        W[k0*n + 0][r] += -1
        W[k1*n + 0][r] += 1
        
        # Term 5: M6
        r = r_base + 5
        U[1*m + 0][r] += 1
        U[0*m + 0][r] += -1
        V[0*p + k0][r] += 1
        V[1*p + k0][r] += 1
        W[k1*n + 1][r] += 1
        
        # Term 6: M7
        r = r_base + 6
        U[0*m + 1][r] += 1
        U[1*m + 1][r] += -1
        V[1*p + k1][r] += 1
        V[0*p + k1][r] += 1
        W[k0*n + 0][r] += 1
        W[k1*n + 1][r] += 1
        
    return {"U": U, "V": V, "W": W}
