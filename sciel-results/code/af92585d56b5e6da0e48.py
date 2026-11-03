"""Matrix multiplication 2,4,8 - Kronecker product decomposition for rank reduction."""
DIMENSIONS = [2, 4, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    U: shape (4, 7), indexed by i*2+j
    V: shape (4, 7), indexed by j*2+k  
    W: shape (4, 7), indexed by k*2+i
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1  # W[k*2+i]: k=0,i=0 -> 0; k=1,i=1 -> 3
    
    # M2: (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1  # k=0,i=1 -> 0*2+1=1
    
    # M3: A00(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=-1  # k=1,i=0 -> 2; k=1,i=1 -> 3
    
    # M4: A11(B10-B00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,0]=-1; W[1,3]=1  # k=0,i=0 -> 0; k=0,i=1 -> 1
    
    # M5: (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[2,4]=1  # k=1,i=0 -> 2
    
    # M6: (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=1  # k=0,i=0 -> 0
    
    # M7: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[3,6]=1  # k=1,i=1 -> 3
    
    return U, V, W


def _identity_1x2x4():
    """Rank decomposition for 1x2x4 matrix multiplication.
    
    A is 1x2, B is 2x4, C is 1x4.
    C_k = A_0*B_0k + A_1*B_1k for k=0,1,2,3
    
    U: shape (1*2, 4) = (2, 4), indexed by i*m+j where i=0, j=0,1
    V: shape (2*4, 4) = (8, 4), indexed by j*p+k where j=0,1, k=0..3
    W: shape (4*1, 4) = (4, 4), indexed by k*n+i where k=0..3, i=0
    
    For each output k, we need 2 terms (for j=0 and j=1):
    Actually, the tensor T[i,j,k] = delta[i,0] * 1 (for all j,k)
    Since i is always 0, T[j,k] = 1 for all j,k.
    
    We can decompose as: for each k, C_k depends on all j.
    Rank 4: one term per output k.
    Term r=k: U[0,r]=1, U[1,r]=1, V[j*4+k, r]=1 for j=0,1, W[k*1+0, r]=1
    """
    U = np.zeros((2, 4), dtype=float)
    V = np.zeros((8, 4), dtype=float)
    W = np.zeros((4, 4), dtype=float)
    
    for k in range(4):
        # U: both j=0 and j=1 contribute
        U[0, k] = 1  # i=0, j=0
        U[1, k] = 1  # i=0, j=1
        # V: for this k, both j=0 and j=1
        V[0*4 + k, k] = 1  # j=0, k
        V[1*4 + k, k] = 1  # j=1, k
        # W: output k, i=0
        W[k*1 + 0, k] = 1
    
    return U, V, W


def _kronecker_decompose(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Combine two decompositions using Kronecker product.
    
    First: (n1 x m1) x (m1 x p1) -> (p1 x n1), rank r1
    Second: (n2 x m2) x (m2 x p2) -> (p2 x n2), rank r2
    
    Result: (n1*n2 x m1*m2) x (m1*m2 x p1*p2) -> (p1*p2 x n1*n2), rank r1*r2
    
    Index conventions:
    U1: (n1*m1, r1), row = i1*m1 + j1
    V1: (m1*p1, r1), row = j1*p1 + k1
    W1: (p1*n1, r1), row = k1*n1 + i1
    
    U2: (n2*m2, r2), row = i2*m2 + j2
    V2: (m2*p2, r2), row = j2*p2 + k2
    W2: (p2*n2, r2), row = k2*n2 + i2
    
    Combined:
    n = n1*n2, m = m1*m2, p = p1*p2
    U: (n*m, r1*r2), row = i*m + j where i = i1*n2 + i2, j = j1*m2 + j2
    V: (m*p, r1*r2), row = j*p + k where j = j1*m2 + j2, k = k1*p2 + k2
    W: (p*n, r1*r2), row = k*n + i where k = k1*p2 + k2, i = i1*n2 + i2
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            # U: for each (i1,j1) in U1 and (i2,j2) in U2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1 = U1[i1 * m1 + j1, c1]
                    if u1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2 = U2[i2 * m2 + j2, c2]
                            if u2 == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j, c] += u1 * u2
            
            # V: for each (j1,k1) in V1 and (j2,k2) in V2
            for j1 in range(m1):
                for k1 in range(p1):
                    v1 = V1[j1 * p1 + k1, c1]
                    if v1 == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2 = V2[j2 * p2 + k2, c2]
                            if v2 == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k, c] += v1 * v2
            
            # W: for each (k1,i1) in W1 and (k2,i2) in W2
            for k1 in range(p1):
                for i1 in range(n1):
                    w1 = W1[k1 * n1 + i1, c1]
                    if w1 == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2 = W2[k2 * n2 + i2, c2]
                            if w2 == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k * n + i, c] += w1 * w2
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Strategy: Kronecker product of Strassen(2x2x2) and Identity(1x2x4)
    # rank = 7 * 4 = 28
    try:
        U1, V1, W1 = _strassen_2x2x2()
        U2, V2, W2 = _identity_1x2x4()
        
        U, V, W = _kronecker_decompose(
            U1, V1, W1, 2, 2, 2,
            U2, V2, W2, 1, 2, 4
        )
        
        rank = U.shape[1]
        
        # Verify correctness
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c] for c in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook rank 64
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
