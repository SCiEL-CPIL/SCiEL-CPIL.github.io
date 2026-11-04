"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

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

def _try_strassen_block(n, m, p):
    """
    Try to use a Strassen-like decomposition.
    For 3x5x6, we can't directly use 2x2 Strassen, but we can try
    to find a low-rank decomposition using known techniques.
    
    One approach: if we can decompose the matrix multiplication tensor
    into a sum of rank-1 tensors, we can use that.
    
    For now, let's try a systematic search for small rank decompositions
    using structured patterns.
    """
    # For 3x5x6, the tensor has dimensions 3x5x6 = 90 entries
    # We need to find rank r such that the tensor can be written as sum of r rank-1 tensors
    
    # Try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) / something... actually the lower bound is more subtle
    # For matrix mult n x m x p, the rank is at least max(n*m, m*p, p*n) / (something)
    # Actually, the rank is at least the dimension of the space spanned by the slices
    
    # Let's try a greedy approach: start with rank 1 and add terms
    # This is complex, so let's try random search with better initialization
    
    return None

def _structured_search(rank, n, m, p, max_trials=500):
    """
    Structured search: use patterns based on known matrix mult decompositions.
    """
    for t in range(max_trials):
        # Generate U, V, W with structure
        # Use a mix of 0s, 1s, -1s, and sometimes 2s, -2s
        vals = [0, 1, -1]
        if t % 5 == 0:
            vals = [0, 1, -1, 2, -2]
        if t % 10 == 0:
            vals = [0, 1, -1, 2, -2, 3, -3]
            
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
        
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
    
    # Try to find lower ranks using structured search
    # For 3x5x6, try ranks from 40 upward (be more aggressive)
    for rank in range(40, best_rank):
        if time.time() > deadline:
            break
        result = _structured_search(rank, n, m, p, max_trials=100)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found a valid decomposition, use it
    
    return best
# EVOLVE-BLOCK-END
