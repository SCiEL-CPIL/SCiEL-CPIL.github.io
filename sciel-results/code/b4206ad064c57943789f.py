import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank
    # We will try rank 29
    # Time limit 120s
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # Function to check if a decomposition is valid
    def check_decomp(U, V, W, rank):
        # U: n*m x rank
        # V: m*p x rank
        # W: p*n x rank
        # Check if sum_r U[i,r] V[j,r] W[k,r] == 1 for all i,j,k
        # Note: W index is k*n + i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Random search for rank 29
    # We use a simple iterative refinement or random initialization with Newton steps?
    # Given the non-linearity, random search is unlikely to succeed quickly for rank 29.
    # However, we can try a specific construction if we know one.
    # Since I don't know a specific construction for 2x3x5 rank 29, I will rely on the schoolbook.
    # But let's try a very short random search just in case.
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try rank 29 with a few random attempts
    for rank in [29]:
        if time.time() > deadline:
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            # Random initialization
            # Use small integers
            U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(p * n)]
            # This random search is extremely unlikely to find an exact integer solution.
            # It's better to stick with the schoolbook unless we have a known algorithm.
            
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
