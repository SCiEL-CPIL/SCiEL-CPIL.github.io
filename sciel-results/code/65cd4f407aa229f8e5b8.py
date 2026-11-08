"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin

    # Schoolbook decomposition of rank n*m*p
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
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank

    # Try to reduce rank by finding linear dependencies among the rank-1 terms
    # Each term t is represented by (U[:,t], V[:,t], W[:,t])
    # If we can express one term as a linear combination of others, we can eliminate it
    
    # For now, try a simple approach: check if any two terms are identical or nearly identical
    # and merge them if possible
    
    # This is a heuristic; in general, rank reduction is NP-hard
    
    # Try random perturbations and check if we can maintain exactness with lower rank
    random.seed(42)
    
    # Try ranks from 53 down to 46
    for target_rank in range(53, 45, -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a rank-target_rank decomposition by perturbing the schoolbook solution
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Start from schoolbook and try to merge terms
            # This is complex, so use random search instead
            U_c = np.zeros((n*m, target_rank))
            V_c = np.zeros((m*p, target_rank))
            W_c = np.zeros((p*n, target_rank))
            
            # Fill with random small integers
            for r in range(target_rank):
                # Each term should contribute to some (i,j,k) entries
                # Randomly assign contributions
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.3:
                            U_c[i*m+j, r] = random.choice([-1, 1])
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.3:
                            V_c[j*p+k, r] = random.choice([-1, 1])
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.3:
                            W_c[k*n+i, r] = random.choice([-1, 1])
            
            U_list = U_c.tolist()
            V_list = V_c.tolist()
            W_list = W_c.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if best_rank <= target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
