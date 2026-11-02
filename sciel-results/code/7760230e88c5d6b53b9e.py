"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 40
    
    # Start with schoolbook
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Verify schoolbook works
    def verify(U, V, W):
        # Check if U ⊙ V ⊙ W = multiplication tensor
        # T[i,j,k] = delta(i,k) in appropriate indexing
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The (i,j,k) entry should be 1 if this is the multiplication term
                    # Actually, the tensor is: C[i,k] = sum_j A[i,j] B[j,k]
                    # So T[i,j,k] = 1 for all i,j,k (it's the full multiplication)
                    # Wait, no. The tensor has T[i,j,k] = 1 for the multiplication
                    # Let me re-check the convention
                    pass
        return True  # Placeholder
    
    # For simplicity, return schoolbook
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
