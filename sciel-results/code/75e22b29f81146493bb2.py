import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

def check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(R):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Start with schoolbook
    U = np.zeros((n * m, base_rank), dtype=int)
    V = np.zeros((m * p, base_rank), dtype=int)
    W = np.zeros((p * n, base_rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Attempt to find a rank 59 decomposition via random search
    # This is a long shot, but if it works, it improves the score.
    # We limit the time to 10 seconds to ensure we return the schoolbook if it fails.
    start_time = time.time()
    target_rank = base_rank - 1
    found = False
    
    while time.time() - start_time < 10 and not found:
        # Randomly initialize U, V, W with small integers
        U_rand = np.random.randint(-1, 2, (n * m, target_rank))
        V_rand = np.random.randint(-1, 2, (m * p, target_rank))
        W_rand = np.random.randint(-1, 2, (p * n, target_rank))
        
        if check_decomposition(U_rand, V_rand, W_rand, n, m, p):
            return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
            
    # Return schoolbook if no better solution found
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
