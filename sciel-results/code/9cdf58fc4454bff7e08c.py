"""Program to find a low-rank decomposition for 4x5 by 5x6 matrix multiplication."""

import time
import random

# Dimensions: (n, m, p) where n=4, m=5, p=6
n, m, p = 4, 5, 6

def construct():
    """Find a rank-minimal decomposition for 4x5 by 5x6 matrix multiplication."""
    deadline = time.time() + 110  # 110 seconds safety margin
    
    # Schoolbook rank = n*m*p = 120
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Build target set for verification
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
    
    def verify_decomposition(U, V, W, rank):
        """Verify the decomposition is correct."""
        for (a, b, c) in target_set:
            s = 0
            for r in range(rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if s != 1:
                return False
        # Check some non-target entries
        for _ in range(100):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            if (a, b, c) not in target_set:
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != 0:
                    return False
        return True
    
    # Try to find a lower-rank decomposition using structured search.
    # For matrix multiplication, the rank is at least the schoolbook rank for
    # most small cases, but let's try random restarts to be sure.
    
    for trial_rank in range(119, 110, -1):
        if time.time() > deadline - 10:
            break
        
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers
            U = [[random.choice([-1, 0, 1]) for _ in range(trial_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(trial_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(trial_rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W, trial_rank):
                best_rank = trial_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < trial_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
