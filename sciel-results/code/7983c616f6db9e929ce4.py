"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    # Try ranks from lb+1 upward (lb itself may not be achievable)
    for rank in range(lb, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Use small integers: -1, 0, 1, or half-integers
            # Initialize U, V, W with random small values
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Try a structured initialization
            for c in range(rank):
                # Randomly assign nonzero entries
                for _ in range(3):
                    ui = random.randint(0, n * m - 1)
                    vi = random.randint(0, m * p - 1)
                    wi = random.randint(0, p * n - 1)
                    val = random.choice([-1, 1])
                    U[ui, c] = val
                    V[vi, c] = val
                    W[wi, c] = val
            
            # Verify
            if _verify(U, V, W, n, m, p):
                # Convert to list of lists with half-integer support
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                # Check if all values are integer or half-integer
                valid = True
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for val in row:
                            if val != int(val) and 2*val != int(2*val):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    if rank < best_rank:
                        best_rank = rank
                        best = {"U": U_list, "V": V_list, "W": W_list}
                    return best
    
    return best

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True
# EVOLVE-BLOCK-END
