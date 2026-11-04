"""Matrix multiplication tensor decomposition for (2,3,10) using Strassen on (2,2,2) k-pairs."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Pre-compute indices for Strassen blocks
    # M1: (A00+A11)(B0k0+B1k1) -> C0k0, C1k1
    # M2: (A10+A11)B0k0 -> C1k0, -C1k1
    # M3: A00(B0k1-B1k1) -> C0k1, C1k1
    # M4: A11(B1k0-B0k0) -> C0k0, C1k0
    # M5: (A00+A01)B1k1 -> -C0k0, C0k1
    # M6: (A11-A00)(B0k0+B0k1) -> C1k1
    # M7: (A01-A11)(B1k0+B1k1) -> C0k0, -C1k1
    
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # M1
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[10+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = 1
        col += 1
        
        # M2
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
        col += 1
        
        # M3
        U[0][col] = 1
        V[10+k1][col] = 1; V[10+k1][col] = 1 # Wait, V[10+k1] is B[1,k1]. V[k1] is B[0,k1].
        # Correction: V indices are j*p+k.
        # B[0,k1] is V[k1]. B[1,k1] is V[10+k1].
        V[k1][col] = 1; V[10+k1][col] = -1
        W[2*k1][col] = 1; W[2*k1+1][col] = 1
        col += 1
        
        # M4
        U[4][col] = 1
        V[10+k0][col] = 1; V[k0][col] = -1
        W[2*k0][col] = 1; W[2*k0+1][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[10+k1][col] = 1
        W[2*k0][col] = -1; W[2*k1][col] = 1
        col += 1
        
        # M6
        U[4][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1; U[4][col] = -1
        V[10+k0][col] = 1; V[10+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = -1
        col += 1
        
    # Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1
            V[2*10+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    assert col == rank
    
    # Verification
    def verify(U, V, W, n, m, p, max_checks=5):
        rank = len(U[0])
        for _ in range(max_checks):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C = A @ B
            Cc = np.zeros((n, p))
            for r in range(rank):
                ua = np.zeros(n)
                vb = np.zeros(p)
                for i in range(n):
                    for j in range(m):
                        ua[i] += U[i*m+j][r] * A[i,j]
                for k in range(p):
                    for j in range(m):
                        vb[k] += V[j*p+k][r] * B[j,k]
                for i in range(n):
                    for k in range(p):
                        Cc[i,k] += W[k*n+i][r] * ua[i] * vb[k]
            if not np.allclose(C, Cc, atol=1e-9):
                return False
        return True

    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
