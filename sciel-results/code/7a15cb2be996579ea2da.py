import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a rank-minimal decomposition using set-cover on intermediate dimension
    # Lower bound: max(n*p, m*p, n*m) = max(14, 21, 6) = 21
    # Try rank = 21 first
    
    # Strategy: For each (j, k) pair, we can potentially share factors across i
    # Since n=2 is small, try to use rank = m*p = 21
    # Each rank term corresponds to a (j, k) pair, and we need to handle both i values
    
    R = m * p  # 21
    
    U_np = np.zeros((n * m, R))
    V_np = np.zeros((m * p, R))
    W_np = np.zeros((p * n, R))
    
    # For each j in [m], k in [p], create a rank term
    # U[i*m+j, r] should be 1 for all i (since A[i,j] multiplies B[j,k])
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1 for all i
    
    r = 0
    for j in range(m):
        for k in range(p):
            # U: all i values for this j
            for i in range(n):
                U_np[i * m + j, r] = 1
            # V: this (j,k)
            V_np[j * p + k, r] = 1
            # W: all i values for this k
            for i in range(n):
                W_np[k * n + i, r] = 1
            r += 1
    
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    # Verify correctness
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In our decomposition: C[i,k] = sum_r U[i*m+j(r), r] * V[j(r)*p+k(r), r] * W[k*n+i, r]
    # For fixed (i,k), this sums over all r where j(r) is anything and k(r)=k
    # For each such r, U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
    # So we get sum over j of 1*1*1 = m terms, but we need sum_j A[i,j]*B[j,k]
    # This doesn't work! The issue is that U and W are constant in i, so we can't distinguish A[i,j] values
    
    # The problem: for fixed (j,k), the term contributes to all (i,k') where k'=k
    # But we need A[i,j]*B[j,k] for each specific i
    
    # This approach fails. Let's go back to trivial but see if we can do better
    
    # Actually, the correct minimal rank for (2,3,7) matrix multiplication is likely 42 (trivial)
    # or possibly less. Let me try the n*p bound: rank = n*p = 14
    # For each (i,k), use one rank term with j summed... no, that doesn't work for exact decomposition
    
    # The trivial decomposition is the safe choice. Let's keep it but verify.
    
    R = n * m * p
    U_np = np.zeros((n * m, R))
    V_np = np.zeros((m * p, R))
    W_np = np.zeros((p * n, R))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_np[i * m + j, r] = 1
                V_np[j * p + k, r] = 1
                W_np[k * n + i, r] = 1
                
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
                
    return {"U": U, "V": V, "W": W}
