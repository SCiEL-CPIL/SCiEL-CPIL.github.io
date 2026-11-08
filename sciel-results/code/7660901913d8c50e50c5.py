import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W

def _assemble_3x5x5_from_3x3x3(U333, V333, W333):
    """
    Assemble 3x5x5 from 3x3x3 decomposition.
    
    A (3x5) split as [A_L (3x2) | A_R (3x3)]
    B (5x5) split as [[B_TL (2x2), B_TR (2x3)],
                       [B_BL (3x2), B_BR (3x3)]]
    
    C = A_L @ B_TL + A_L @ B_TR + A_R @ B_BL + A_R @ B_BR
    """
    n, m, p = 3, 5, 5
    
    R1 = 3*2*2  # 12
    R2 = 3*2*3  # 18
    R3 = 3*3*2  # 18
    R4 = U333.shape[1]
    
    total = R1 + R2 + R3 + R4
    U = np.zeros((n*m, total))
    V = np.zeros((m*p, total))
    W = np.zeros((p*n, total))
    
    off = 0
    # Block 1: A_L[i,j] (j=0,1) * B_TL[j',k] (j'=0,1; k=0,1)
    for i in range(n):
        for j in range(2):
            for k in range(2):
                U[i*m+j, off] = 1.0
                V[j*p+k, off] = 1.0
                W[k*n+i, off] = 1.0
                off += 1
    
    # Block 2: A_L[i,j] (j=0,1) * B_TR[j',k] (j'=0,1; k=2,3,4)
    for i in range(n):
        for j in range(2):
            for k in range(2, 5):
                U[i*m+j, off] = 1.0
                V[j*p+k, off] = 1.0
                W[k*n+i, off] = 1.0
                off += 1
    
    # Block 3: A_R[i,j] (j=2,3,4) * B_BL[j',k] (j'=2,3,4; k=0,1)
    for i in range(n):
        for j in range(2, 5):
            for k in range(2):
                U[i*m+j, off] = 1.0
                V[j*p+k, off] = 1.0
                W[k*n+i, off] = 1.0
                off += 1
    
    # Block 4: A_R[i,j] (j=2,3,4) * B_BR[j',k] (j'=2,3,4; k=2,3,4)
    # Local indices: jl = j-2, jl' = j'-2, kl = k-2
    for c in range(R4):
        for i in range(n):
            for jl in range(3):
                U[i*m + (jl+2), off+c] = U333[i*3 + jl, c]
        for jl in range(3):
            for kl in range(3):
                V[(jl+2)*p + (kl+2), off+c] = V333[jl*3 + kl, c]
        for kl in range(3):
            for i in range(n):
                W[(kl+2)*n + i, off+c] = W333[kl*3 + i, c]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Use schoolbook 3x3x3 for guaranteed correctness
    U333, V333, W333 = _schoolbook(3, 3, 3)
    
    U, V, W = _assemble_3x5x5_from_3x3x3(U333, V333, W333)
    
    if not _verify(U, V, W):
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
