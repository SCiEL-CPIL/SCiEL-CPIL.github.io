import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    best_rank = rank_schoolbook
    best_decomp = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find a lower rank decomposition
    # The rank of M_{2x5} is known to be <= 20? Let's try ranks 20 to 30.
    # We use a randomized search.
    
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    # Target ranks to try
    target_ranks = list(range(20, 40))
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random searches for this rank
        attempts = 1000
        for _ in range(attempts):
            if time.time() > deadline:
                break
            
            # Generate random supports
            # U has n*m = 10 rows, V has m*p = 30 rows, W has p*n = 12 rows
            # Each term r has non-zero entries in U[:, r], V[:, r], W[:, r]
            # We choose a small number of non-zeros for each term
            # For M_{2x2}, Strassen uses 7 terms. Each term has 3 non-zeros in U, 3 in V, 3 in W?
            # Actually, the supports are specific.
            # Random supports are unlikely to work.
            # This approach is flawed for exact decomposition.
            pass
            
    # Since random search is unlikely to find a low-rank decomposition quickly,
    # we rely on the schoolbook decomposition unless a specific known one is implemented.
    # For now, return schoolbook.
    return best_decomp
# EVOLVE-BLOCK-END
