"""Program for matrix multiplication 3,4,7 tensor decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _build_strassen_decomposition():
    n, m, p = DIMENSIONS
    # We will construct U, V, W as lists of columns
    # Each column corresponds to one rank-1 term
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        U = np.zeros(n * m)
        V = np.zeros(m * p)
        W = np.zeros(p * n)
        U[i * m + j] = 1
        V[j * p + k] = 1
        W[k * n + i] = 1
        U_cols.append(U)
        V_cols.append(V)
        W_cols.append(W)
        
    # Helper to add a Strassen block for I, J, K
    # I, J, K are tuples of 2 indices
    def add_strassen(I, J, K):
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # Strassen's 7 products
        # M1 = (A[i1,j1] + A[i2,j1]) * (B[j1,k1] + B[j2,k1]) ? 
        # Standard Strassen for 2x2:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+b)(e+f)  -> (A[0,0]+A[0,1]) * (B[0,0]+B[1,0])? No.
        # Standard layout:
        # A = [[a, b], [c, d]]
        # B = [[e, f], [g, h]]
        # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
        # M1 = (a+b)(e+f)
        # M2 = (c+d)(e+f)
        # M3 = a(f-g)
        # M4 = d(h-f)
        # M5 = (a+d)(h-e)
        # M6 = (b+c)(e+h)
        # M7 = (b+d)(g+h)
        
        # Mapping to our indices:
        # a = A[i1, j1], b = A[i1, j2]
        # c = A[i2, j1], d = A[i2, j2]
        # e = B[j1, k1], f = B[j1, k2]
        # g = B[j2, k1], h = B[j2, k2]
        
        # M1 = (a+b)(e+f)
        # U: A[i1,j1] + A[i1,j2] -> U[i1*m+j1]=1, U[i1*m+j2]=1
        # V: B[j1,k1] + B[j1,k2] -> V[j1*p+k1]=1, V[j1*p+k2]=1
        # W: C[i1,k1] contribution? 
        # C[i1,k1] = M1 + M4 - M5 + M7
        # C[i1,k2] = M3 + M5
        # C[i2,k1] = M2 + M4
        # C[i2,k2] = M1 - M2 + M3 + M6
        
        # We need to add 7 columns.
        
        # M1
        U1 = np.zeros(n*m); V1 = np.zeros(m*p); W1 = np.zeros(p*n)
        U1[i1*m+j1] = 1; U1[i1*m+j2] = 1
        V1[j1*p+k1] = 1; V1[j1*p+k2] = 1
        W1[k1*n+i1] = 1; W1[k2*n+i2] = 1 # Wait, W1 contributes to C[i1,k1] and C[i2,k2]
        # Let's check the W coefficients carefully.
        # C[i1,k1] gets M1. So W1[k1*n+i1] += 1.
        # C[i2,k2] gets M1. So W1[k2*n+i2] += 1.
        # Is that correct?
        # C[i1,k1] = M1 + M4 - M5 + M7
        # C[i2,k2] = M1 - M2 + M3 + M6
        # Yes, M1 appears in both.
        
        # M2
        U2 = np.zeros(n*m); V2 = np.zeros(m*p); W2 = np.zeros(p*n)
        U2[i2*m+j1] = 1; U2[i2*m+j2] = 1
        V2[j1*p+k1] = 1; V2[j1*p+k2] = 1
        W2[k1*n+i2] = 1 # C[i2,k1] gets M2
        W2[k2*n+i2] = -1 # C[i2,k2] gets -M2
        
        # M3
        U3 = np.zeros(n*m); V3 = np.zeros(m*p); W3 = np.zeros(p*n)
        U3[i1*m+j1] = 1
        V3[j1*p+k2] = 1; V3[j2*p+k1] = -1
        W3[k2*n+i1] = 1 # C[i1,k2] gets M3
        W3[k2*n+i2] = 1 # C[i2,k2] gets M3
        
        # M4
        U4 = np.zeros(n*m); V4 = np.zeros(m*p); W4 = np.zeros(p*n)
        U4[i2*m+j2] = 1
        V4[j2*p+k2] = 1; V4[j1*p+k2] = -1
        W4[k1*n+i1] = 1 # C[i1,k1] gets M4
        W4[k1*n+i2] = 1 # C[i2,k1] gets M4
        
        # M5
        U5 = np.zeros(n*m); V5 = np.zeros(m*p); W5 = np.zeros(p*n)
        U5[i1*m+j1] = 1; U5[i2*m+j2] = 1
        V5[j2*p+k2] = 1; V5[j1*p+k1] = -1
        W5[k1*n+i1] = -1 # C[i1,k1] gets -M5
        W5[k2*n+i1] = 1  # C[i1,k2] gets M5
        
        # M6
        U6 = np.zeros(n*m); V6 = np.zeros(m*p); W6 = np.zeros(p*n)
        U6[i1*m+j2] = 1; U6[i2*m+j1] = 1
        V6[j1*p+k1] = 1; V6[j2*p+k2] = 1
        W6[k2*n+i2] = 1 # C[i2,k2] gets M6
        
        # M7
        U7 = np.zeros(n*m); V7 = np.zeros(m*p); W7 = np.zeros(p*n)
        U7[i1*m+j2] = 1; U7[i2*m+j2] = 1
        V7[j2*p+k1] = 1; V7[j2*p+k2] = 1
        W7[k1*n+i1] = 1 # C[i1,k1] gets M7
        W7[k2*n+i2] = 1 # C[i2,k2] gets M7
        
        U_cols.extend([U1, U2, U3, U4, U5, U6, U7])
        V_cols.extend([V1, V2, V3, V4, V5, V6, V7])
        W_cols.extend([W1, W2, W3, W4, W5, W6, W7])

    # Define the 6 Strassen blocks
    blocks = [
        ((0, 1), (0, 1), (0, 1)),
        ((0, 1), (0, 1), (2, 3)),
        ((0, 1), (0, 1), (4, 5)),
        ((0, 1), (2, 3), (0, 1)),
        ((0, 1), (2, 3), (2, 3)),
        ((0, 1), (2, 3), (4, 5)),
    ]
    
    # Track which (i,j,k) tuples are covered by Strassen
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
                    add_strassen(I, J, K) if False else None # Placeholder, call below
    
    # Actually, add_strassen adds 7 terms. We call it once per block.
    # But my code above calls add_strassen inside the loop? No, I defined it.
    # I need to call it for each block.
    
    # Re-initialize lists
    U_cols = []
    V_cols = []
    W_cols = []
    covered = set()
    
    for I, J, K in blocks:
        add_strassen(I, J, K)
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
                    
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)
                    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

_cached_result = None

def construct():
    global _cached_result
    if _cached_result is None:
        _cached_result = _build_strassen_decomposition()
    return _cached_result
