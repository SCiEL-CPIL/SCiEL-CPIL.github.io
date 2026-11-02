"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Naive decomposition as fallback
    def naive_decomposition():
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
    
    best_result = naive_decomposition()
    best_rank = n * m * p
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try block decomposition: split middle dim as 3+2, last dim as 3+3
    # Blocks: (2x3)*(3x3), (2x3)*(3x3), (2x2)*(2x3), (2x2)*(2x3)
    # For each block, use naive decomposition (will be same as overall naive)
    # This gives no improvement, so let's try to find a better sub-decomposition
    
    # Attempt: use a rank-7 construction for (2x3)*(3x3) if we can build one
    # For now, just return naive if we can't find better in time
    
    if time.time() > deadline:
        return best_result
    
    # Try to improve by attempting a simple heuristic: 
    # Start from naive, try to set some entries to 0 and see if still valid
    # This is very heuristic and unlikely to work well, but let's try a bit
    
    U = [row[:] for row in best_result["U"]]
    V = [row[:] for row in best_result["V"]]
    W = [row[:] for row in best_result["W"]]
    rank = best_rank
    
    # Try to zero out columns that might be redundant
    # Check if any column is a linear combination of others
    # This is expensive, so only try if we have time
    
    if time.time() < deadline - 30:
        # Try random perturbations and see if we can reduce rank
        import random
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U_try = [row[:] for row in U]
            V_try = [row[:] for row in V]
            W_try = [row[:] for row in W]
            
            # Try to zero out one random entry in each of U, V, W for a random column
            col_to_try = random.randint(0, rank - 1)
            
            # Find a non-zero entry in U for this column
            u_idx = random.randint(0, n * m - 1)
            if U_try[u_idx][col_to_try] != 0:
                U_try[u_idx][col_to_try] = 0
                V_try[random.randint(0, m * p - 1)][col_to_try] = 0
                W_try[random.randint(0, p * n - 1)][col_to_try] = 0
                
                if verify(U_try, V_try, W_try):
                    U, V, W = U_try, V_try, W_try
                    best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
