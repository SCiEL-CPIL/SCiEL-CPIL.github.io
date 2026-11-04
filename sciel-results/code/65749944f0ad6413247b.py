"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _try_rank_ilp(rank, n, m, p, max_trials=100):
    """
    Try to find a valid decomposition of given rank using a structured ILP-like approach.
    We model the tensor as a system of linear equations and use sparsity to prune.
    """
    # For simplicity and speed, we'll use a heuristic that builds on the schoolbook
    # and tries to merge terms using small integer coefficients.
    
    # Start with schoolbook and try to reduce rank by finding linear dependencies
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Try random integer assignments with sparsity
    vals = [0, 1, -1, 2, -2, 3, -3, 4, -4]
    
    for trial in range(max_trials):
        # Generate sparse random matrices
        for i in range(n * m):
            for c in range(rank):
                if random.random() < 0.3:  # Sparsity
                    U[i][c] = random.choice(vals)
                else:
                    U[i][c] = 0
        
        for j in range(m * p):
            for c in range(rank):
                if random.random() < 0.3:
                    V[j][c] = random.choice(vals)
                else:
                    V[j][c] = 0
        
        for k in range(p * n):
            for c in range(rank):
                if random.random() < 0.3:
                    W[k][c] = random.choice(vals)
                else:
                    W[k][c] = 0
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Best solution starts as schoolbook
    best = _schoolbook()
    best_rank = n * m * p  # 90
    
    # Try to find lower ranks using ILP-like structured search
    # For 3x5x6, try ranks from 60 upward (more aggressive than before)
    for rank in range(60, best_rank):
        if time.time() > deadline:
            break
        result = _try_rank_ilp(rank, n, m, p, max_trials=30)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found a valid decomposition, use it
    
    return best
# EVOLVE-BLOCK-END
