"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # U: (n*m) x rank = 6 x rank
    # V: (m*p) x rank = 18 x rank  
    # W: (p*n) x rank = 12 x rank
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 36
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # Try Kronecker product approach
    # 6 = 2*3, so try to exploit this structure
    # Decompose as tensor product of (2x3)x(3x1) and (1x1)x(1x3)?
    # Or try direct low-rank search
    
    # First, try a structured approach based on the 6=2*3 factorization
    # Represent the 3x6 matrix B as having a Kronecker structure
    # B[i, k*3+l] where i in 0..2, k in 0..1, l in 0..2
    # C[i, k*3+l] = sum_j A[i,j] * B[j, k*3+l]
    
    # If we can write B[j, k*3+l] = sum over some decomposition...
    
    # Let's try rank 24, 20, 18, etc.
    for target_rank in [18, 20, 24, 27, 30, 33]:
        if time.time() > deadline:
            break
            
        # Try random initialization with structured hints
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            rank = target_rank
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Verify: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] should equal 1
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list format
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = rank
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # If no better found, return schoolbook
    if best_rank == schoolbook_rank:
        rank = schoolbook_rank
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
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
