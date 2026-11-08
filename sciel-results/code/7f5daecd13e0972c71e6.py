"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Deadline guard: keep a safety margin before the 120s limit
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    def _time_remaining():
        return deadline - time.time()
    
    def _verify(U, V, W):
        """
        Verify that U, V, W exactly decompose the matrix multiplication tensor.
        The tensor T[i, j, k] for (i in range(n), j in range(m), k in range(p))
        should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0  # The multiplication tensor has 1s for valid (i,j,k)
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def _make_schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Start with schoolbook as fallback
    best = _make_schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: For n x m x p matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # Here: max(8, 20, 10) = 20
    # Upper bound from schoolbook: 40
    # Let's try ranks from 20 upward
    
    # Strategy: Use structured search with random restarts
    # The tensor T has entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try a few different approaches:
    
    # Approach 1: Try rank = 20, 21, ... with structured patterns
    # For small dimensions, we can try systematic search
    
    # First, let's try a simple structured approach for low ranks
    # The key insight: each term in the sum for C[i,k] involves one A[i,j] and one B[j,k]
    # So we need to "cover" all (i,j,k) triples with rank-many terms
    
    # Let's try rank from 20 up to 40
    for target_rank in range(20, min(41, best_rank)):
        if _time_remaining() < 5.0:
            break
        
        # Try random search for this rank
        found = _try_random_search(target_rank, n, m, p, nm, mp, pn, max_iters=5000, time_limit=3.0)
        if found is not None:
            U, V, W = found
            if _verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # We found the minimal rank, stop
    
    # If we found rank 20, that's optimal (lower bound)
    # Otherwise, return the best we found
    return best


def _try_random_search(target_rank, n, m, p, nm, mp, pn, max_iters=5000, time_limit=3.0):
    """
    Try to find a decomposition of the given rank using random search.
    Returns (U, V, W) if found, None otherwise.
    """
    start_time = time.time()
    
    # Use integer entries for simplicity (halves would be more general but let's start with ints)
    # Actually, let's allow small integers: -2, -1, 0, 1, 2
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random U, V, W with small integer entries
        U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(nm)]
        V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(mp)]
        W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(pn)]
        
        # Verify
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        valid = False
                        break
        
        if valid:
            return U, V, W
    
    return None
# EVOLVE-BLOCK-END
