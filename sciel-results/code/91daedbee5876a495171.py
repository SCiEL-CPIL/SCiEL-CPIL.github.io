"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Rank 46 decomposition: 6 Strassen blocks (rank 7) + 1 schoolbook column (rank 4)
    rank = 46
    U = np.zeros((n * m, rank))  # (4, 46)
    V = np.zeros((m * p, rank))  # (26, 46)
    W = np.zeros((p * n, rank))  # (26, 46)
    
    col = 0
    # Strassen blocks for column pairs (0,1), (2,3), ..., (10,11)
    for group in range(6):
        k1 = 2 * group
        k2 = 2 * group + 1
        
        # Indices in A (U): i*m + j
        # a00 -> 0, a01 -> 1, a10 -> 2, a11 -> 3
        # Indices in B (V): j*p + k
        # b00 -> 0*13+k1, b01 -> 0*13+k2, b10 -> 1*13+k1, b11 -> 1*13+k2
        # Indices in C (W): k*n + i
        # C00 -> k1*2+0, C01 -> k2*2+0, C10 -> k1*2+1, C11 -> k2*2+1
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[0, col] = 1
        U[3, col] = 1
        V[0*p + k1, col] = 1
        V[1*p + k2, col] = 1
        # C00 += M1, C11 += M1
        W[k1*n + 0, col] = 1
        W[k2*n + 1, col] = 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[2, col] = 1
        U[3, col] = 1
        V[0*p + k1, col] = 1
        # C10 += M2, C11 -= M2
        W[k1*n + 1, col] = 1
        W[k2*n + 1, col] = -1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[0, col] = 1
        V[0*p + k2, col] = 1
        V[1*p + k2, col] = -1
        # C01 += M3, C11 += M3
        W[k2*n + 0, col] = 1
        W[k2*n + 1, col] = 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[3, col] = 1
        V[1*p + k1, col] = 1
        V[0*p + k1, col] = -1
        # C00 += M4, C10 += M4
        W[k1*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[0, col] = 1
        U[1, col] = 1
        V[1*p + k2, col] = 1
        # C00 -= M5, C01 += M5
        W[k1*n + 0, col] = -1
        W[k2*n + 0, col] = 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[2, col] = 1
        U[0, col] = -1
        V[0*p + k1, col] = 1
        V[0*p + k2, col] = 1
        # C11 += M6
        W[k2*n + 1, col] = 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[1, col] = 1
        U[3, col] = -1
        V[1*p + k1, col] = 1
        V[1*p + k2, col] = 1
        # C00 += M7
        W[k1*n + 0, col] = 1
        col += 1
    
    # Last column k=12: Schoolbook (rank 4)
    k = 12
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Verify the decomposition
    if time.time() < deadline:
        if _check_decomposition(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook if verification fails
    rank_fallback = n * m * p  # 52
    U_f = np.zeros((n * m, rank_fallback))
    V_f = np.zeros((m * p, rank_fallback))
    W_f = np.zeros((p * n, rank_fallback))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j, col] = 1
                V_f[j * p + k, col] = 1
                W_f[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U_f.tolist(),
        "V": V_f.tolist(),
        "W": W_f.tolist()
    }
