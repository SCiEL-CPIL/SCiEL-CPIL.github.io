"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank and try to reduce
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find better decomposition using structured approach
    # For 4x4 x 4x7, try rank values
    # Use a simple heuristic: try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    
    # Try a few candidate ranks
    candidate_ranks = [28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        # Try to construct a decomposition of this rank
        # Use a simple approach: random initialization + local search
        # But for exactness, we need integer/half-integer entries
        
        # For now, just use schoolbook if we can't find better
        # This is a placeholder - in a real implementation, we'd search
        pass
    
    return best_result
# EVOLVE-BLOCK-END
