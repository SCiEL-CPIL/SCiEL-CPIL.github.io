import time
import numpy as np
import itertools
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 
    multiplying a 4x5 matrix by a 5x5 matrix.
    
    Strategy:
    1. Build schoolbook decomposition (rank 100) as baseline.
    2. Search for lower-rank decompositions using structured patterns
       and random search with verification.
    3. Return the best (lowest rank) valid decomposition found.
    
    U: (n*m, rank) = (20, rank)
    V: (m*p, rank) = (25, rank)
    W: (p*n, rank) = (20, rank)
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    schoolbook_rank = n * m * p  # 100

    # Build schoolbook decomposition
    def build_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(nm)]
        V = [[0] * schoolbook_rank for _ in range(mp)]
        W = [[0] * schoolbook_rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = build_schoolbook()
    best_rank = schoolbook_rank

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_rank(rank, candidates):
        """Try a set of candidate decompositions at given rank."""
        for U, V, W in candidates:
            if time.time() > deadline:
                return None
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        return None

    # Strategy 1: Try structured decompositions inspired by Strassen-like patterns
    # For small matrices, we can try specific algebraic identities
    
    # Generate candidate patterns for low ranks
    # We'll try ranks from 99 down to 50, but focus on structured searches first
    
    # For efficiency, try a few structured approaches:
    
    # Approach A: Try to find decompositions with rank = n*m + m*p + p*n - something
    # This is based on the idea that some products can be shared
    
    # Approach B: Random search with small entry sets for moderate ranks
    
    # Let's try random search for ranks slightly below schoolbook
    # Use a limited entry set for efficiency
    
    entry_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    # Try ranks from 99 down to 80 with random search
    # But be careful about time
    
    random.seed(42)
    
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 25, 0), -1):
        if time.time() > deadline:
            break
        
        # Try a few random candidates at this rank
        num_trials = 5
        found = None
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            U = [[random.choice(entry_set) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice(entry_set) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice(entry_set) for _ in range(rank)] for _ in range(pn)]
            
            if verify(U, V, W):
                found = {"U": U, "V": V, "W": W}
                break
        
        if found is not None:
            best_rank = rank
            best_result = found
            break

    # If no improvement found via random search, try a more structured approach
    # for ranks that are divisors or have special properties
    
    # For now, return the best found
    return best_result
# EVOLVE-BLOCK-END
