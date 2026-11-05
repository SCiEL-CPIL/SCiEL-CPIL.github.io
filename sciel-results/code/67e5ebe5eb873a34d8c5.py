"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a better decomposition using random search
    # We will try ranks from 71 down to 60
    for target_rank in range(rank_schoolbook - 1, 59, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few random attempts for this rank
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Generate random U, V, W with integer/half-integer entries
            # We use small integers to keep the numbers manageable
            U = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check if the decomposition is valid
            # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # T[i, j, k] = 1 for all i, j, k
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break # Found a valid decomposition for this rank, try lower rank
        
        if best_rank < target_rank + 1:
            continue # Found a better rank, continue to next lower rank
        else:
            # If we didn't find a valid decomposition for this rank, we can stop
            # But we might find one with a different random seed
            # So we continue to next lower rank
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
