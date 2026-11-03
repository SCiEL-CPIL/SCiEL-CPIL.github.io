"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # First, return schoolbook as a valid fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, column] = 1
                V_sb[j * p + k, column] = 1
                W_sb[k * n + i, column] = 1
                column += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Try to find a lower-rank decomposition using rank-1 tensor updates
    # We'll use a greedy approach: try to find rank-1 terms that reduce the residual
    
    # The matrix multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # T has entries T[i,j,k] = 1 for all i,j,k (it's the identity in the tensor sense)
    # Actually, the tensor is T_{i,j,k} where the multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k
    
    # Let's work with the tensor T of shape (n, m, p) where T[i,j,k] = 1
    # A rank-r decomposition is: T[i,j,k] = sum_{t=1}^{r} U[i,j,t] * V[j,k,t] * W[k,i,t]
    # (with appropriate index conventions)
    
    # Using the trace convention: W[k*n+i, r]
    # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Let's try a random search for rank-1 terms
    # Start with full tensor and try to subtract rank-1 tensors
    
    # For efficiency, let's work with a specific structure
    # Try ranks from 60 down to 50
    
    for target_rank in range(70, 55, -1):
        if time.time() > deadline - 20:
            break
            
        # Try to find a decomposition of this rank
        # Use a simple approach: start with random rank-1 terms and adjust
        
        # For now, let's just verify if the schoolbook is optimal
        # or try a specific known decomposition
        
        # The 3x5x5 case: known bounds suggest rank could be lower than 75
        # Let's try a structured approach based on polynomial multiplication
        
        # Represent 5x5 matrix multiplication using cyclic convolution
        # A 5x5 matrix multiplication can be decomposed using CRT
        # x^5 - 1 = (x-1)(x^4+x^3+x^2+x+1) over rationals
        # Over extension fields, we can factor further
        
        # For now, return the best we have
        pass
    
    return best_result
# EVOLVE-BLOCK-END
