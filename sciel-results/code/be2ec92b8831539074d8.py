"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the standard rank-45 decomposition."""
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

def _verify_full(U, V, W):
    """Full verification - check all (i,j,k) triples exactly."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _try_structured_rank(target_rank, time_budget):
    """
    Try to find a decomposition of the given target_rank using a structured ansatz.
    
    Ansatz: Use a block structure where we group terms by j-index.
    For each j, we have a 3x5 sub-problem. We try to share computations.
    
    Strategy: Assume U[i*m+j, c] has a form that allows sharing.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + time_budget
    
    # Try a specific structured approach:
    # For the 3x3 x 3x5 case, try rank 30 by using 10 terms per j-group
    # with some sharing across i and k.
    
    # Alternative: try to use a Kronecker-like structure
    # U[i,j,c] = a[i, c//3] * b[j, c%3] for some factorization
    # This might not work directly, but let's try random search with structure.
    
    # Simple random search with integer/half-integer entries
    # For small ranks, this might work.
    
    r = target_rank
    if r < 15:  # Below the trivial lower bound max(n*m, m*p, p*n) = 15
        return None
    
    # Try a few random initializations with small integer values
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Initialize with small random integers or half-integers
        U = [[random.choice([-1, 0, 0.5, 1]) for _ in range(r)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 0.5, 1]) for _ in range(r)] for _ in range(m*p)]
        W = [[random.choice([-1, 0, 0.5, 1]) for _ in range(r)] for _ in range(p*n)]
        
        if _verify_full(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_rank_reduction():
    """
    Try to find a decomposition with rank less than 45.
    Use known algebraic decompositions and structured search.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110  # Leave 10s margin
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Try ranks from 15 up to 44
    
    # First, try some known structured decompositions
    # For 3x3 x 3x5, one approach is to use the fact that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can try to group the j terms.
    
    # Try rank 30: use 10 terms per j-index (3 j's * 10 = 30)
    # This would require each j-group to handle a 3x5 product in rank 10.
    # A 3x5 matrix-vector product can be done in rank 15 (schoolbook) or less.
    
    # Let's try a different structure: 
    # Use the identity that for fixed j, the contribution is A[:,j] * B[j,:]
    # which is an outer product of a 3-vector and a 5-vector.
    # If we can share the "A[:,j]" part across different k values, we save terms.
    
    # Actually, the standard decomposition already does this implicitly.
    # The key to reducing rank is to find algebraic dependencies.
    
    # Try a specific construction:
    # For 3x3 x 3x5, try to use rank 36 by combining some terms.
    
    # Let's try a systematic approach: for each candidate rank, 
    # set up the polynomial system and try to solve it.
    
    # Given time constraints, try a few specific ranks with random search.
    for target_rank in [44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 5:
            break
        
        # Try random search for this rank
        result = _try_structured_rank(target_rank, min(10, deadline - time.time()))
        if result is not None:
            return result
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    
    Returns a dictionary with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Safe fallback: schoolbook decomposition
    best = _schoolbook()
    best_rank = n * m * p  # 45
    
    # Try to find a lower-rank decomposition
    result = _try_rank_reduction()
    if result is not None:
        # Verify the result
        if _verify_full(result["U"], result["V"], result["W"]):
            return result
    
    # If we ran out of time or couldn't find a better decomposition, return schoolbook
    return best
# EVOLVE-BLOCK-END
