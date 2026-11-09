"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Generate schoolbook decomposition as baseline
    def schoolbook():
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using structured search
    # Key insight: use Kronecker product structure and rank-1 updates
    
    # For matrix multiplication, the tensor is T[i,j,k] = 1 for all (i,j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try various rank values from high to low
    for target_rank in range(best_rank - 1, 40, -1):
        if time.time() > deadline - 20:
            break
        
        # Use structured random search with half-integer values
        found = False
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Generate random matrices with half-integer entries
            vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            U_rand = np.random.choice(vals, size=(n*m, target_rank))
            V_rand = np.random.choice(vals, size=(m*p, target_rank))
            W_rand = np.random.choice(vals, size=(p*n, target_rank))
            
            # Verify the decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_rand[i*m+j, :] * V_rand[j*p+k, :] * W_rand[k*n+i, :])
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                found = True
                break
        
        if found:
            # If we found a decomposition at this rank, try to go lower
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
