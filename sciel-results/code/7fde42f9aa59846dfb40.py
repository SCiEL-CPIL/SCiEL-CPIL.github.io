import numpy as np

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Helper to add Strassen terms for a 2x2x2 block starting at column k
    def add_strassen(k):
        nonlocal col
        # Strassen coefficients for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[k][col] += 1; V[k+3][col] += 1 # B00 is at index k*1+0? No.
        # V index: j*p + k.
        # B is 2x2 block. Indices (0, k), (1, k), (0, k+1), (1, k+1).
        # V[0*p+k] = V[k]. V[1*p+k] = V[p+k].
        # V[0*p+k+1] = V[k+1]. V[1*p+k+1] = V[p+k+1].
        
        # Let's map indices correctly.
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # B indices: 0:(0,k), 1:(0,k+1), 2:(1,k), 3:(1,k+1).
        # C indices: 0:(k,0), 1:(k,1), 2:(k+1,0), 3:(k+1,1).
        # W index: i*n + j.
        # W[k*2+0], W[k*2+1], W[k*2+2], W[k*2+3].
        
        # M1 = (A00+A11)(B00+B11) -> (U[0]+U[3]) (V[0]+V[2])
        U[0][col] += 1; U[3][col] += 1
        V[k][col] += 1; V[p+k][col] += 1
        W[k*n][col] += 1; W[k*n+1][col] += 1; W[k*n+2][col] += 1; W[k*n+3][col] += 1
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1?
        # No, Strassen combination is complex.
        
        # I will use a verified Strassen implementation.
        # P1 = (a+d)(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[k][col] = 1; V[p+k][col] = 1
        W[k*n][col] = 1; W[k*n+1][col] = -1; W[k*n+2][col] = -1; W[k*n+3][col] = 1
        col += 1
        
        # P2 = (c+d)B10
        U[2][col] = 1; U[3][col] = 1
        V[p+k][col] = 1
        W[k*n+2][col] = 1; W[k*n+3][col] = 1
        col += 1
        
        # P3 = a(B01-B11)
        U[0][col] = 1
        V[k+1][col] = 1; V[p+k+1][col] = -1
        W[k*n][col] = 1; W[k*n+1][col] = 1; W[k*n+2][col] = 1; W[k*n+3][col] = 1
        col += 1
        
        # P4 = d(B10-B00)
        U[3][col] = 1
        V[p+k][col] = 1; V[k][col] = -1
        W[k*n][col] = 1; W[k*n+1][col] = -1; W[k*n+2][col] = 1; W[k*n+3][col] = -1
        col += 1
        
        # P5 = (a+b)B11
        U[0][col] = 1; U[1][col] = 1
        V[p+k+1][col] = 1
        W[k*n+1][col] = 1; W[k*n+3][col] = 1
        col += 1
        
        # P6 = (c-a)(B00+B10)
        U[2][col] = 1; U[0][col] = -1
        V[k][col] = 1; V[p+k][col] = 1
        W[k*n][col] = 1; W[k*n+2][col] = 1
        col += 1
        
        # P7 = (b-d)(B01+B11)
        U[1][col] = 1; U[3][col] = -1
        V[k+1][col] = 1; V[p+k+1][col] = 1
        W[k*n+1][col] = 1; W[k*n+3][col] = 1
        col += 1

    # Apply Strassen for cols 0,1; 2,3; 4,5
    add_strassen(0)
    add_strassen(2)
    add_strassen(4)
    
    # Schoolbook for col 6
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+6][col] = 1
            W[6*n+i][col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
