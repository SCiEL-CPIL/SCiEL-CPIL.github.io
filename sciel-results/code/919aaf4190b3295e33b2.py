import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def _schoolbook(n, m, p):
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

    def _validate(U, V, W, n, m, p):
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Check all valid indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Try to find a rank < 45 decomposition
    # Strategy: Try rank 44, 43, ... down to a limit, using random search with integer constraints
    
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # We will try to find a rank 44 decomposition.
    # If we find one, we return it.
    
    # Given the difficulty, we might not find one.
    # But let's try a few random restarts.
    
    rng = random.Random(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
            
        rank = best_rank - 1
        if rank < 15: # Lower bound
            break
            
        # Random initialization with small integers
        U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        
        if _validate(U, V, W, n, m, p):
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            # If we found a better one, we can continue searching for even lower rank
            # But usually, if we find 44, we might try 43 next loop if we decrement rank
            
    return best_result
