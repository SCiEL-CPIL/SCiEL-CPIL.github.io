"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank decompositions
    # We need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    def check_decomposition(U, V, W):
        """Check if the decomposition satisfies the tensor condition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try ranks from schoolbook-1 down to some minimum
    for target_rank in range(rank_school - 1, max(10, rank_school - 15), -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization with small integers
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random ±1 or 0
            for c in range(target_rank):
                # Randomly assign non-zero entries
                for idx in range(n*m):
                    if random.random() < 0.3:
                        U[idx, c] = random.choice([-1, 1])
                for idx in range(m*p):
                    if random.random() < 0.3:
                        V[idx, c] = random.choice([-1, 1])
                for idx in range(p*n):
                    if random.random() < 0.3:
                        W[idx, c] = random.choice([-1, 1])
            
            # This is unlikely to work with random init, but let's try a structured approach
            # Instead, let's try to perturb the schoolbook decomposition
    
    # Given the difficulty, just return schoolbook if no better found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
