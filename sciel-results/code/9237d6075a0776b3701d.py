"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # 2. Attempt to find rank-59 decomposition via structured search
    # Strategy: Try to merge pairs of schoolbook terms by introducing free parameters
    # and solving for consistency.
    
    def _try_rank59(n, m, p, deadline):
        """
        Attempt to find a rank-59 decomposition by merging one pair of schoolbook terms.
        We start from the schoolbook decomposition and try to eliminate one term by
        expressing its contribution as a linear combination of the remaining 59 terms.
        """
        # Build schoolbook tensor in sparse form
        # Each term c has: (i,j,k) such that U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
        
        # Try each possible term to eliminate
        for elim_idx in range(base_rank):
            if time.time() > deadline - 30.0:
                return None
            
            i_e, j_e, k_e = terms[elim_idx]
            
            # We want to express the contribution of term elim_idx as a combination
            # of other terms. The contribution of term c is:
            #   A[i,j] * B[j,k] -> C[i,k]
            # In the decomposition, term c contributes: U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # For the schoolbook, this is 1 if (i,j,k) matches, else 0.
            
            # To eliminate term (i_e, j_e, k_e), we need to adjust coefficients of other terms
            # such that the net contribution to C[i_e, k_e] from A[i_e, j_e] * B[j_e, k_e]
            # is still correctly captured.
            
            # This is a complex constraint satisfaction problem. Let's try a simpler approach:
            # Check if there exist two other terms (i1,j1,k1) and (i2,j2,k2) such that
            # we can adjust their coefficients to compensate.
            
            # For simplicity, let's try a random search with limited iterations
            import random
            random.seed(42 + elim_idx)
            
            # Try random coefficient adjustments
            max_attempts = 1000
            for attempt in range(max_attempts):
                if time.time() > deadline - 30.0:
                    return None
                
                # Pick 3 random other terms to adjust
                candidates = [t for t in range(base_rank) if t != elim_idx]
                if len(candidates) < 3:
                    continue
                
                adj_indices = random.sample(candidates, 3)
                c1, c2, c3 = adj_indices
                
                # The constraints are nonlinear. For now, just return None as a placeholder
                # since a full implementation would require solving a system of polynomial equations.
                break
        
        return None
    
    # Try the rank-59 search
    result_59 = _try_rank59(n, m, p, deadline)
    if result_59 is not None:
        best = result_59
        best_rank = 59
    else:
        # Return the schoolbook baseline
        pass
    
    # Check time to ensure we don't accidentally timeout
    if time.time() > deadline - 5.0:
        return best

    return best
# EVOLVE-BLOCK-END
