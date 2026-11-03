"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

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

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    # Check that off-diagonal terms are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            total = 0
                            for r in range(rank):
                                total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                            if total != 0:
                                return False
    return True

def _try_random_rank(n, m, p, target_rank, seed=0, max_iters=500):
    """Try to find a decomposition of the given target rank."""
    rng = np.random.default_rng(seed)
    
    for it in range(max_iters):
        # Generate random factor matrices
        U = rng.integers(-2, 3, size=(n * m, target_rank))
        V = rng.integers(-2, 3, size=(m * p, target_rank))
        W = rng.integers(-2, 3, size=(p * n, target_rank))
        
        # Convert to lists for verification
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_structured_search(n, m, p, target_rank, deadline):
    """Try structured approaches to find lower rank decompositions."""
    # For small matrices, we can try various patterns
    
    # Try using the property that matrix multiplication tensor has certain symmetries
    # For 2x4x6, the Strassen-like algorithms might help
    
    # One approach: block decomposition
    # Split the 4x6 matrix into blocks and use efficient multiplication
    
    # For now, try a few specific constructions
    # Construction 1: Use Kronecker product structure
    # 2x4 = 2x(2x2), 4x6 = (2x2)x(2x3)
    
    # Try rank = 2*2*2*2*3 = 48 (same as schoolbook, no improvement)
    # But we can try to optimize the 2x2x2 part using Strassen (rank 7 instead of 8)
    # That would give 7*2*3 = 42
    
    # Let's try a Strassen-like approach
    # 2x2x2 matrix multiplication has rank 7 (Strassen)
    # So for 2x4x6, we can view it as:
    # (2x2) @ (2x2) @ (2x3) with some blocking
    
    # Actually, let's think differently:
    # 2x4 matrix @ 4x6 matrix
    # = (2x2 | 2x2) @ (2x2 @ 2x3)
    # This is getting complex. Let's try random search with more iterations.
    
    best_result = None
    for seed in range(20):
        if time.time() > deadline:
            break
        result = _try_random_rank(n, m, p, target_rank, seed=seed, max_iters=100)
        if result is not None:
            best_result = result
            break
    
    return best_result

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Leave safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Known results: For 2x4x6, the rank is known to be less than 48
    
    # Try various target ranks from low to high
    for target_rank in range(24, best_rank):
        if time.time() > deadline - 10:  # Leave 10s margin
            break
        
        result = _try_structured_search(n, m, p, target_rank, deadline)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
