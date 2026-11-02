"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_result = None
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, max(best_rank - 10, 40), -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values in {-1, 0, 1}
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Set some entries to random values
            for c in range(target_rank):
                # Randomly assign non-zero entries
                for _ in range(3):
                    i1 = random.randint(0, n*m-1)
                    i2 = random.randint(0, m*p-1)
                    i3 = random.randint(0, p*n-1)
                    val = random.choice([-1, 1])
                    U[i1, c] = val
                    V[i2, c] = val
                    W[i3, c] = val
            
            # Check if this is a valid decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(target_rank):
                            total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break  # Found a valid decomposition for this rank
        
        if best_result is not None:
            break  # Found a lower rank, stop
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
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
    
    return best_result
# EVOLVE-BLOCK-END
