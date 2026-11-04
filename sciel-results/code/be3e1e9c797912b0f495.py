import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x6)
    # A1 uses A[i, 0] for i in 0,1
    # B1 uses B[0, k] for k in 0..5
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x6)
    # A2 uses A[i, 1], A[i, 2]
    # B2 uses B[1, k], B[2, k]
    
    # Strassen factors for 2x2x2
    # U_strassen (2x2, 7), V_strassen (2x2, 7), W_strassen (2x2, 7)
    # Indices in local 2x2x2:
    # U: (i, j) where i,j in 0,1
    # V: (j, k) where j,k in 0,1
    # W: (k, i) where k,i in 0,1
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We will define the local factors and then map them.
    
    for t in range(3):
        # Block t corresponds to B columns 2t, 2t+1
        # Local B indices 0, 1 map to global B columns 2t, 2t+1
        # Local B rows 0, 1 map to global B rows 1, 2
        
        # Local A indices 0, 1 map to global A cols 1, 2
        # Local A rows 0, 1 map to global A rows 0, 1
        
        # Local C indices 0, 1 map to global C cols 2t, 2t+1
        # Local C rows 0, 1 map to global C rows 0, 1
        
        # Define local Strassen factors
        # U_local[i, j]
        # V_local[j, k]
        # W_local[k, i]
        
        # M1
        U[0*m + 1, col] += 1.0
        U[1*m + 2, col] += 1.0
        V[1*p + 2*t, col] += 1.0
        V[2*p + 2*t + 1, col] += 1.0
        W[2*t*n + 0, col] += 1.0
        W[(2*t + 1)*n + 1, col] += 1.0
        col += 1
        
        # M2
        U[1*m + 1, col] += 1.0
        U[1*m + 2, col] += 1.0
        V[1*p + 2*t, col] += 1.0
        W[2*t*n + 1, col] += 1.0
        W[(2*t + 1)*n + 1, col] += -1.0
        col += 1
        
        # M3
        U[0*m + 1, col] += 1.0
        V[1*p + 2*t + 1, col] += 1.0
        V[2*p + 2*t + 1, col] += -1.0
        W[(2*t + 1)*n + 0, col] += 1.0
        W[(2*t + 1)*n + 1, col] += 1.0
        col += 1
        
        # M4
        U[1*m + 2, col] += 1.0
        V[2*p + 2*t, col] += 1.0
        V[1*p + 2*t, col] += -1.0
        W[2*t*n + 0, col] += 1.0
        W[2*t*n + 1, col] += 1.0
        col += 1
        
        # M5
        U[0*m + 1, col] += 1.0
        U[0*m + 2, col] += 1.0
        V[2*p + 2*t + 1, col] += 1.0
        W[2*t*n + 0, col] += -1.0
        W[(2*t + 1)*n + 0, col] += 1.0
        col += 1
        
        # M6
        U[1*m + 1, col] += 1.0
        U[0*m + 1, col] += -1.0
        V[1*p + 2*t, col] += 1.0
        V[1*p + 2*t + 1, col] += 1.0
        W[(2*t + 1)*n + 1, col] += 1.0
        col += 1
        
        # M7
        U[0*m + 2, col] += 1.0
        U[1*m + 2, col] += -1.0
        V[2*p + 2*t, col] += 1.0
        V[2*p + 2*t + 1, col] += 1.0
        W[2*t*n + 0, col] += 1.0
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
