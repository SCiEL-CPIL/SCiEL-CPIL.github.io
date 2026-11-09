"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x4 * 4x5.
    Uses a systematic search for lower-rank decompositions.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # The tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B 
    # contribute to (i,k) entry of C
    # For our case: n=2, m=4, p=5
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (20, rank) 
    # W has shape (p*n, rank) = (10, rank)
    
    # Start with schoolbook rank and try to reduce
    best_rank = n * m * p  # 40
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook solution first as baseline
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank decompositions using random search
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # where valid means the product contributes to C[i,k]
    
    # For each rank from 20 up to 39, try to find a decomposition
    for target_rank in range(20, best_rank):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                    break
                    
            np.random.seed(attempt * 1000 + target_rank)
            
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Check if this random decomposition works
            # The tensor equation: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
            # if (i,j,k) is a valid triple, else 0
            
            valid = True
            # Check all possible (i,j,k) combinations
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # This triple is valid - it should sum to 1
                        val = np.sum(U_rand[i*m+j, :] * V_rand[j*p+k, :] * W_rand[k*n+i, :])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                continue
                
            # Found a valid decomposition!
            best_rank = target_rank
            best_U = U_rand
            best_V = V_rand
            best_W = W_rand
            break  # Found valid decomposition for this rank
            
        # If we found a valid decomposition for this rank, stop
        if best_rank == target_rank:
            break
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
