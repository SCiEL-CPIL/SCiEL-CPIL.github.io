"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    N = n * m  # 6
    M = m * p  # 27
    P = p * n  # 18
    
    rank_schoolbook = n * m * p  # 54
    
    U = np.zeros((N, rank_schoolbook), dtype=int)
    V = np.zeros((M, rank_schoolbook), dtype=int)
    W = np.zeros((P, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using random search
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all i,j,k
    
    # For small dimensions, we can try random integer/half-integer matrices
    # and check if they decompose the tensor
    
    best_rank = rank_schoolbook
    best_decomp = None
    
    # Try ranks from 27 (lower bound for some cases) up to schoolbook
    # The theoretical lower bound is max(n*m, m*p, p*n) = 27
    
    # For 2x3x9, the minimum rank is known to be 27 (same as max dimension product)
    # Let's try to find a rank-27 decomposition
    
    # Actually, for matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # Here that's max(6, 27, 18) = 27
    
    # Let's try a structured approach: use the fact that we can share computations
    # across different k values for the same (i,j)
    
    # A rank-27 decomposition would require that each of the 27 (j,k) pairs
    # corresponds to exactly one rank term, and we share across i
    
    # Try: for each (j,k), have a rank term that handles all i
    # U[i*m+j, r] should depend on i, V[j*p+k, r] = 1, W[k*n+i, r] should depend on i
    # This doesn't work directly because U and W both depend on i
    
    # Alternative: use the identity that 2x3 matrix mult by 3x9 can be done
    # by computing 2x3 * 3x9 = sum over j of (column j of A) * (row j of B)
    # That's 3 terms, but each term is outer product of 2-vectors and 9-vectors
    # Rank would be 3 * 2 * 9 / 2 = 27? No, that's not right.
    
    # Let's just try random search for rank 27
    np.random.seed(42)
    
    for target_rank in [27, 28, 29, 30]:
        if time.time() > deadline - 5:
            break
        
        # Try random initialization with small integers
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.zeros((N, target_rank))
            V_rand = np.zeros((M, target_rank))
            W_rand = np.zeros((P, target_rank))
            
            # Initialize with small random integers
            for r in range(target_rank):
                for idx in range(N):
                    U_rand[idx, r] = np.random.randint(-2, 3)
                for idx in range(M):
                    V_rand[idx, r] = np.random.randint(-2, 3)
                for idx in range(P):
                    W_rand[idx, r] = np.random.randint(-2, 3)
            
            # Check if this decomposes the tensor
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We need this to equal 1 for all i,j,k
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_decomp = (U_rand, V_rand, W_rand)
                break
        
        if best_decomp is not None:
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
