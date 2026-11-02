import time
import numpy as np

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Block structure:
    - j=0: schoolbook 2x1x5, rank 10
    - j in {1,2}, k in {0,1}: Strassen 2x2x2, rank 7
    - j in {1,2}, k=2: schoolbook 2x2x1, rank 4
    - j in {1,2}, k in {3,4}: Strassen 2x2x2, rank 7
    Total: 28
    
    Indexing convention:
    - U[i*m+j, r]: coefficient for A[i,j] in factor r
    - V[j*p+k, r]: coefficient for B[j,k] in factor r
    - W[k*n+i, r]: coefficient for output C[i,k] in factor r
    """
    n, m, p = 2, 3, 5
    R = 28
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    
    # Block A: j=0, all k -> schoolbook 2x1x5
    for i in range(2):
        for k in range(5):
            U[i * m, r] = 1.0
            V[k, r] = 1.0
            W[k * n + i, r] = 1.0
            r += 1
    # r = 10
    
    # Helper: Strassen 2x2x2 for rows i=0,1; cols j=base_j, base_j+1; k=base_k, base_k+1
    def strassen_222(bj, bk):
        nonlocal r
        # M1: (A0b + A1(b+1)) * (Bb,k + B(b+1),k+1)
        U[0*m + bj, r] = 1.0
        U[1*m + bj+1, r] = 1.0
        V[bj*p + bk, r] = 1.0
        V[(bj+1)*p + bk+1, r] = 1.0
        W[bk*n, r] = 1.0
        W[(bk+1)*n + 1, r] = 1.0
        r += 1
        
        # M2: (A1b + A1(b+1)) * Bb,k
        U[1*m + bj, r] = 1.0
        U[1*m + bj+1, r] = 1.0
        V[bj*p + bk, r] = 1.0
        W[bk*n + 1, r] = 1.0
        W[(bk+1)*n + 1, r] = -1.0
        r += 1
        
        # M3: A0b * (Bb,k+1 - B(b+1),k+1)
        U[0*m + bj, r] = 1.0
        V[bj*p + bk+1, r] = 1.0
        V[(bj+1)*p + bk+1, r] = -1.0
        W[(bk+1)*n, r] = 1.0
        W[(bk+1)*n + 1, r] = 1.0
        r += 1
        
        # M4: A1(b+1) * (B(b+1),k - Bb,k)
        U[1*m + bj+1, r] = 1.0
        V[(bj+1)*p + bk, r] = 1.0
        V[bj*p + bk, r] = -1.0
        W[bk*n, r] = 1.0
        W[bk*n + 1, r] = 1.0
        r += 1
        
        # M5: (A0b + A0(b+1)) * B(b+1),k+1
        U[0*m + bj, r] = 1.0
        U[0*m + bj+1, r] = 1.0
        V[(bj+1)*p + bk+1, r] = 1.0
        W[bk*n, r] = -1.0
        W[(bk+1)*n, r] = 1.0
        r += 1
        
        # M6: (A1b - A0b) * (Bb,k + Bb,k+1)
        U[1*m + bj, r] = 1.0
        U[0*m + bj, r] = -1.0
        V[bj*p + bk, r] = 1.0
        V[bj*p + bk+1, r] = 1.0
        W[(bk+1)*n + 1, r] = 1.0
        r += 1
        
        # M7: (A0(b+1) - A1(b+1)) * (B(b+1),k + B(b+1),k+1)
        U[0*m + bj+1, r] = 1.0
        U[1*m + bj+1, r] = -1.0
        V[(bj+1)*p + bk, r] = 1.0
        V[(bj+1)*p + bk+1, r] = 1.0
        W[bk*n, r] = 1.0
        r += 1
    
    # Block B: j=1,2; k=0,1 -> Strassen
    strassen_222(1, 0)
    # r = 17
    
    # Block C: j=1,2; k=2 -> schoolbook 2x2x1
    for i in range(2):
        for jj in range(1, 3):
            U[i*m + jj, r] = 1.0
            V[jj*p + 2, r] = 1.0
            W[2*n + i, r] = 1.0
            r += 1
    # r = 21
    
    # Block D: j=1,2; k=3,4 -> Strassen
    strassen_222(1, 3)
    # r = 28
    
    assert r == R, f"Rank mismatch: got {r}, expected {R}"
    
    # Convert to list of lists with integer values
    # Ensure we handle potential floating point precision issues, though entries are exactly +/-1 or 0
    Ul = [[int(round(float(x))) for x in row] for row in U]
    Vl = [[int(round(float(x))) for x in row] for row in V]
    Wl = [[int(round(float(x))) for x in row] for row in W]
    
    return {"U": Ul, "V": Vl, "W": Wl}
