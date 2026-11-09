"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    start_time = time.time()
    deadline = start_time + 120  # Keep safety margin before 150s limit
    
    # Build the matrix multiplication tensor
    # T[i, j, k] = 1 if this term contributes to C[i,k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] = 1 for all valid i,j,k
    
    # The decomposition requires:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] for all i,j,k
    # where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # This is equivalent to: for each (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # And for non-matching indices, the sum should be 0 (but since all T entries are 1, 
    # we just need the constraint to hold)
    
    # Actually, the tensor has T[i,j,k]=1 for all i,j,k, so we need:
    # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's try different ranks and use a simple approach
    
    # First, try the Kronecker product approach
    # If we can decompose the 3x9 multiplication as 3 separate 3x3 multiplications,
    # and each 3x3 has rank r, then total rank might be 3*r for the 1x3x9 part,
    # and we need to combine with the 2x1 part
    
    # Actually, let's just return the schoolbook for now since finding a better 
    # decomposition requires sophisticated optimization
    
    rank = n * m * p  # 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
