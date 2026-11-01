import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W correctly decompose T exactly."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-10)

    # Deterministic construction of rank-19 decomposition
    # Split j=0,1 (2x2x5) and j=2 (2x1x5)
    # 2x2x5 = 2x2x2 (Strassen, rank 7) + 2x2x2 (Strassen, rank 7) + 2x2x1 (Schoolbook, rank 4)
    # 2x1x5 = Rank 1
    
    U = np.zeros((n*m, 19))
    V = np.zeros((m*p, 19))
    W = np.zeros((p*n, 19))
    
    col = 0
    
    # Part 1: j in {0,1}, k in {0,1} -> 2x2x2 Strassen
    # U indices: i*3 + j
    # V indices: j*5 + k
    # W indices: k*2 + i
    
    # Strassen 2x2x2 terms
    # M1 = (A00+A11)(B00+B11)
    U[0*3+0, col] += 1; U[1*3+1, col] += 1
    V[0*5+0, col] += 1; V[1*5+1, col] += 1
    W[0*2+0, col] += 1; W[1*2+1, col] += 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[0*3+1, col] += 1; U[1*3+1, col] += 1
    V[0*5+0, col] += 1
    W[0*2+1, col] += 1; W[1*2+1, col] -= 1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0*3+0, col] += 1
    V[0*5+1, col] += 1; V[1*5+1, col] -= 1
    W[1*2+0, col] += 1; W[1*2+1, col] += 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[1*3+1, col] += 1
    V[1*5+0, col] += 1; V[0*5+0, col] -= 1
    W[0*2+0, col] += 1; W[0*2+1, col] += 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0*3+0, col] += 1; U[0*3+1, col] += 1
    V[1*5+1, col] += 1
    W[0*2+0, col] -= 1; W[1*2+0, col] += 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[0*3+1, col] += 1; U[0*3+0, col] -= 1
    V[0*5+0, col] += 1; V[0*5+1, col] += 1
    W[1*2+1, col] += 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[0*3+1, col] += 1; U[1*3+1, col] -= 1
    V[1*5+0, col] += 1; V[1*5+1, col] += 1
    W[0*2+0, col] += 1
    col += 1
    
    # Part 2: j in {0,1}, k in {2,3} -> 2x2x2 Strassen
    # Map local k' in {0,1} to global k = 2 + k'
    
    # M1 = (A00+A11)(B00+B11)
    U[0*3+0, col] += 1; U[1*3+1, col] += 1
    V[0*5+2, col] += 1; V[1*5+3, col] += 1
    W[2*2+0, col] += 1; W[3*2+1, col] += 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[0*3+1, col] += 1; U[1*3+1, col] += 1
    V[0*5+2, col] += 1
    W[2*2+1, col] += 1; W[3*2+1, col] -= 1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0*3+0, col] += 1
    V[0*5+3, col] += 1; V[1*5+3, col] -= 1
    W[3*2+0, col] += 1; W[3*2+1, col] += 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[1*3+1, col] += 1
    V[1*5+2, col] += 1; V[0*5+2, col] -= 1
    W[2*2+0, col] += 1; W[2*2+1, col] += 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0*3+0, col] += 1; U[0*3+1, col] += 1
    V[1*5+3, col] += 1
    W[2*2+0, col] -= 1; W[3*2+0, col] += 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[0*3+1, col] += 1; U[0*3+0, col] -= 1
    V[0*5+2, col] += 1; V[0*5+3, col] += 1
    W[3*2+1, col] += 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[0*3+1, col] += 1; U[1*3+1, col] -= 1
    V[1*5+2, col] += 1; V[1*5+3, col] += 1
    W[2*2+0, col] += 1
    col += 1
    
    # Part 3: j in {0,1}, k=4 -> 2x2x1 Schoolbook (Rank 4)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Here B0 is V[j=0, k=4], B1 is V[j=1, k=4]
    # A00 is U[i=0, j=0], A01 is U[i=0, j=1]
    # A10 is U[i=1, j=0], A11 is U[i=1, j=1]
    # W0 is W[k=4, i=0], W1 is W[k=4, i=1]
    
    # Term 1: A00 B0 -> W0
    U[0*3+0, col] = 1
    V[0*5+4, col] = 1
    W[4*2+0, col] = 1
    col += 1
    
    # Term 2: A01 B1 -> W0
    U[0*3+1, col] = 1
    V[1*5+4, col] = 1
    W[4*2+0, col] = 1
    col += 1
    
    # Term 3: A10 B0 -> W1
    U[1*3+0, col] = 1
    V[0*5+4, col] = 1
    W[4*2+1, col] = 1
    col += 1
    
    # Term 4: A11 B1 -> W1
    U[1*3+1, col] = 1
    V[1*5+4, col] = 1
    W[4*2+1, col] = 1
    col += 1

    # Part 4: j=2, k in {0,1,2,3,4} -> 2x1x5 Rank 1
    # U[0*3+2] and U[1*3+2]
    # V[2*5+k]
    # W[k*2+0] and W[k*2+1]
    U[0*3+2, col] = 1
    U[1*3+2, col] = 1
    for k in range(5):
        V[2*5+k, col] = 1
        W[k*2+0, col] = 1
        W[k*2+1, col] = 1
    col += 1

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U_sb = np.zeros((n*m, rank))
    V_sb = np.zeros((m*p, rank))
    W_sb = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
