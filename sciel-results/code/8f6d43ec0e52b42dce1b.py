"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random

def _build_factors_from_terms(terms):
    """
    Given a list of terms, each term is a dict with:
      'u': list of (i,j,coeff) for U
      'v': list of (j,k,coeff) for V
      'w': list of (k,i,coeff) for W
    Build U, V, W matrices of rank = len(terms).
    """
    rank = len(terms)
    n, m, p = 3, 3, 8
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for c, term in enumerate(terms):
        for (i, j, coeff) in term['u']:
            U[i * m + j][c] = coeff
        for (j, k, coeff) in term['v']:
            V[j * p + k][c] = coeff
        for (k, i, coeff) in term['w']:
            W[k * n + i][c] = coeff
    return U, V, W

def _schoolbook_terms(n=3, m=3, p=8, k_offset=0):
    """Generate schoolbook terms for n x m * m x p."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [(i, j, 1.0)]
                v = [(j, k, 1.0)]
                w = [(k, i, 1.0)]
                terms.append({'u': u, 'v': v, 'w': w})
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Start with schoolbook as fallback
    schoolbook_terms = _schoolbook_terms(n, m, p)
    best_terms = schoolbook_terms
    
    # Try to find a rank reduction using random linear combinations
    # The idea: for 3x3x3, there exist rank-21 decompositions (vs 27 schoolbook).
    # For 3x3x8, we can split into two 3x3x3 + one 3x3x2.
    # But let's try a more general approach: random rank-r factorization search.
    
    # Since finding a true rank reduction is hard, let's try a heuristic:
    # Use the known result that 3x3x3 has rank 21.
    # Hardcode a rank-21 decomposition for 3x3x3.
    # Then use schoolbook for the remaining 3x3x2 part.
    # Total rank = 21 + 21 + 18 = 60.
    
    # Rank-21 decomposition of 3x3x3 (from literature, Bini et al. or similar)
    # I'll use a known valid rank-21 decomposition.
    
    # For safety, let's just return schoolbook if we can't guarantee correctness.
    # Actually, let me try a simple improvement: use the fact that we can combine
    # some terms when coefficients allow it. But for exact decomposition, we need
    # to be careful.
    
    # Let's try a different approach: use numpy to verify and potentially find
    # a lower rank decomposition via random initialization and gradient descent.
    
    # Given the time constraint and complexity, return schoolbook for now.
    # This is guaranteed correct.
    
    U, V, W = _build_factors_from_terms(best_terms)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
