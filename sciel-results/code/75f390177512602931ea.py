"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    Uses schoolbook as baseline and attempts rank reduction via structured search.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    schoolbook_rank = n * m * p  # 90
    
    # Deadline guard: leave 5s margin before 150s kill
    start_time = time.time()
    deadline = start_time + 140.0
    
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    # Try to find lower-rank decompositions
    # Strategy: exploit dimension asymmetry, try ranks just below schoolbook
    
    def try_structured_decomposition(rank, max_iters=500):
        """Try to find a valid decomposition of given rank using structured random search."""
        # Use a pool of small integer/half-integer values
        val_pool = [-1, -0.5, 0, 0.5, 1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with sparse random half-integers
            U = [[random.choice(val_pool) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice(val_pool) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice(val_pool) for _ in range(rank)] for _ in range(pn)]
            
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        return None
    
    # Candidate ranks to try, from highest to lowest
    # Focus on ranks near schoolbook that might be achievable
    candidate_ranks = list(range(schoolbook_rank - 1, max(schoolbook_rank - 15, 70), -1))
    
    search_start = time.time()
    max_search_time = 120.0
    
    for rank in candidate_ranks:
        if time.time() > deadline or (time.time() - search_start) > max_search_time:
            break
        
        # Use more iterations for higher ranks (closer to schoolbook, more likely to work)
        iters = 200 if rank > schoolbook_rank - 5 else 100
        
        result = try_structured_decomposition(rank, max_iters=iters)
        if result is not None:
            best = result
            # Found a lower rank, continue trying to go lower
            if time.time() > deadline - 15:
                break
    
    # Final verification
    if not verify(best["U"], best["V"], best["W"], len(best["U"][0])):
        # Fallback to schoolbook
        best = schoolbook()
    
    return best
# EVOLVE-BLOCK-END
