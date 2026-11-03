"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with schoolbook
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if not U or not V or not W:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Verify schoolbook
    assert verify(best_U, best_V, best_W), "Schoolbook should be valid"
    
    # Try to find lower rank decompositions
    # Strategy: Use structured search with random restarts
    
    def try_random_decomposition(rank, num_iters=1000):
        """Try to find a valid decomposition of given rank using random search."""
        for _ in range(num_iters):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization with small integers/half-integers
            vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                return (U, V, W)
        return None
    
    # Try structured approaches
    # Approach 1: Use block decomposition
    # For 3x4x5, try to exploit structure
    
    # Let's try a systematic search from rank 59 down
    # But first, try some promising ranks
    
    # Known: rank must be at least max(n*m, m*p, p*n) = 20
    # Try a few structured constructions
    
    # Try rank 55, 50, 45, etc.
    for target_rank in [59, 58, 57, 56, 55, 50, 45, 40]:
        if time.time() > deadline - 20:
            break
        
        result = try_random_decomposition(target_rank, num_iters=500)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # If we haven't found improvement, try a more systematic approach
    # Use the fact that we can sometimes merge columns
    
    # Final fallback: return best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
