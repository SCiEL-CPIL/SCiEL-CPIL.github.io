import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # For each block b in 0, 1, 2
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11
        U[0*m+0][7*b+0] = 1
        U[1*m+1][7*b+0] = 1
        # V: B00, B11
        V[0*p+k0][7*b+0] = 1
        V[1*p+k1][7*b+0] = 1
        # W: C00, C11
        W[k0*n+0][7*b+0] = 1
        W[k1*n+1][7*b+0] = 1
        
        # M2: (A10+A11)B00
        U[1*m+0][7*b+1] = 1
        U[1*m+1][7*b+1] = 1
        V[0*p+k0][7*b+1] = 1
        W[k0*n+1][7*b+1] = 1 # C10
        
        # M3: A00(B01-B11)
        U[0*m+0][7*b+2] = 1
        V[0*p+k1][7*b+2] = 1
        V[1*p+k1][7*b+2] = -1
        W[k1*n+0][7*b+2] = 1 # C01
        W[k1*n+1][7*b+2] = 1 # C11
        
        # M4: A11(B10-B00)
        U[1*m+1][7*b+3] = 1
        V[1*p+k0][7*b+3] = 1
        V[0*p+k0][7*b+3] = -1
        W[k0*n+0][7*b+3] = 1 # C00
        W[k0*n+1][7*b+3] = 1 # C10
        
        # M5: (A00+A10)B11
        U[0*m+0][7*b+4] = 1
        U[1*m+0][7*b+4] = 1
        V[1*p+k1][7*b+4] = 1
        W[k1*n+0][7*b+4] = 1 # C01
        W[k0*n+0][7*b+4] = -1 # C00
        
        # M6: (A11-A00)(B00+B01)
        U[1*m+1][7*b+5] = 1
        U[0*m+0][7*b+5] = -1
        V[0*p+k0][7*b+5] = 1
        V[0*p+k1][7*b+5] = 1
        W[k1*n+1][7*b+5] = 1 # C11
        
        # M7: (A01-A11)(B10+B11)
        U[0*m+1][7*b+6] = 1
        U[1*m+1][7*b+6] = -1
        V[1*p+k0][7*b+6] = 1
        V[1*p+k1][7*b+6] = 1
        W[k0*n+0][7*b+6] = 1 # C00
        
    return {"U": U, "V": V, "W": W}
