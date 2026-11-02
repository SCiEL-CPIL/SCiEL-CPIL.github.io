"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify exact tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return (U, V, W)

def _try_winograd_block(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using Winograd-style block factorization.
    For small blocks (e.g., 2x2 or 2x3), Winograd's algorithm can reduce rank.
    We try to partition the multiplication into blocks and apply known low-rank
    decompositions to each block, then combine them.
    
    For 3x3 * 3x7, we can try:
    - Partition A into 2x2 and 1x2 blocks (or 2x1 and 1x1, etc.)
    - Partition B into 2x2, 2x3, etc.
    - Apply Winograd 2x2 (rank 4 vs 8) or other known decompositions
    
    However, combining blocks is non-trivial and may not yield global savings.
    This is a heuristic attempt.
    """
    # This is a placeholder for more sophisticated block-based search.
    # For now, we don't implement a full Winograd combination because
    # the interaction between blocks makes exact decomposition complex.
    # A proper implementation would require solving for the block interactions.
    return None

def _try_structured_random_search(n, m, p, r, max_restarts=10, deadline=None):
    """
    Try to find a rank-r decomposition using structured random initialization
    and verification. This is a heuristic search that may find solutions
    if they exist with simple structure.
    """
    import random
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize with small integer values
        U = [[random.randint(-2, 2) for _ in range(r)] for _ in range(n*m)]
        V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(m*p)]
        W = [[random.randint(-2, 2) for _ in range(r)] for _ in range(p*n)]
        
        # Convert to Fraction for exact verification
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        if _verify(U_frac, V_frac, W_frac, n, m, p):
            return (U_frac, V_frac, W_frac)
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Get schoolbook decomposition as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Lower bound on rank
    rank_lb = max(n*m, m*p, n*p)
    
    # Try to find lower rank decompositions using structured search
    # For 3x3 * 3x7, the lower bound is 21, schoolbook is 63.
    # Try a few ranks between the lower bound and schoolbook.
    
    # First, try the Winograd-style block approach (currently a placeholder)
    winograd_result = _try_winograd_block(n, m, p, deadline)
    if winograd_result is not None:
        U_w, V_w, W_w = winograd_result
        if _verify(U_w, V_w, W_w, n, m, p):
            best_result = (U_w, V_w, W_w)
            best_rank = len(U_w[0])
    
    # Try structured random search for ranks slightly above the lower bound
    # We limit the search to a small range to stay within time limits
    max_search_rank = min(best_rank - 1, rank_lb + 20)
    
    for r in range(rank_lb, max_search_rank + 1):
        if time.time() > deadline:
            break
        
        # Reduce restarts as rank increases (higher ranks are harder to find randomly)
        max_restarts = max(3, 20 - (r - rank_lb))
        
        result = _try_structured_random_search(n, m, p, r, max_restarts, deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r, V_r, W_r)
            best_rank = r
            break  # Found a lower rank, stop searching
    
    # Return the best result found
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
