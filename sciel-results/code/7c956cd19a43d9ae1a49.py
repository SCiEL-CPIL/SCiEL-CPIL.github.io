import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    n, m, p = 2, 4, 7
    # Dimensions:
    # A is n x m (2 x 4)
    # B is m x p (4 x 7)
    # C is n x p (2 x 7)
    
    # Rank R
    # Schoolbook rank is n*m*p = 2*4*7 = 56.
    # We can potentially do better, but let's start with a valid decomposition.
    # A simple Kronecker product approach or just iterating over all terms.
    
    R = n * m * p
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((n*p, R), dtype=int)
    
    # Indices:
    # A[i, j] -> U index i*m + j
    # B[j, k] -> V index j*p + k
    # C[i, k] -> W index k*n + i (as per contract: W[k*n+i, r])
    
    # The tensor T[i, j, k] = C[i, k] = sum_j A[i, j] * B[j, k]
    # Decomposition: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the schoolbook algorithm, each term in the sum corresponds to a specific (i, j, k).
    # Actually, the standard decomposition has one rank-1 term for each (i, j, k) triple?
    # No, the rank is the number of scalar multiplications.
    # In schoolbook, we perform n*m*p multiplications.
    # Each multiplication is A[i, j] * B[j, k].
    # This contributes to C[i, k].
    # So for each (i, j, k), we have a rank-1 component:
    # U has 1 at (i, j)
    # V has 1 at (j, k)
    # W has 1 at (i, k)
    
    # Let's verify the index mapping for W.
    # Contract says: W[k*n+i, r]
    # So for C[i, k], the index in W is k*n + i.
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j] corresponds to row i*m + j
                U[i*m + j, idx] = 1
                
                # V[j, k] corresponds to row j*p + k
                V[j*p + k, idx] = 1
                
                # W[i, k] corresponds to row k*n + i
                W[k*n + i, idx] = 1
                
                idx += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if idx < R:
        # If we timed out (unlikely for R=56), return what we have?
        # But the evaluator expects valid shapes.
        # For R=56, this loop is instantaneous.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
