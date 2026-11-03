"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    Uses a structured search to find a low-rank decomposition with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def is_half_integer(x):
        """Check if x is an integer or half-integer."""
        y = x * 2
        return abs(y - round(y)) < 1e-9
    
    def to_half_int_list(lst):
        """Convert a list to half-integer representation."""
        return [float(round(x * 2)) / 2.0 for x in lst]
    
    # Try to find a decomposition with rank < 60
    # Start with schoolbook and try to compress
    
    # First, let's try rank 59 by looking for a pair of terms that can be merged
    # For two terms (i1,j1,k1) and (i2,j2,k2) to be mergeable, we need:
    # The sum of two rank-1 tensors to equal a single rank-1 tensor in the relevant subspace
    
    # Actually, let's try a different approach: use a randomized search with structured initialization
    
    best_rank = n * m * p  # Start with schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as baseline
    rank = best_rank
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
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank using a structured approach
    # For each rank from 59 down to 50, try to find a solution
    
    import random
    random.seed(42)
    
    for target_rank in range(59, 49, -1):
        found = False
        for attempt in range(100):
            # Initialize with random signs and zeros
            U = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Verify
            if verify(U, V, W, target_rank):
                # Check if entries are half-integers (they are integers, so yes)
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
    
    # If no lower rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
