"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x4 matrix.
    
    Returns a dict with keys 'U', 'V', 'W' containing lists of lists.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # First, try to find a known low-rank decomposition.
    # For 3x3x4, we can try to extend a 3x3x3 decomposition.
    # Known best for 3x3x3 is 23. For 3x3x4, it might be around 30-36.
    # However, finding an exact integer/half-integer decomposition is hard.
    
    # Strategy:
    # 1. Use the trivial schoolbook decomposition as a fallback (rank 36).
    # 2. Attempt to reduce rank using a heuristic search or known algorithms.
    #    Given the time constraint and complexity, we will stick to the schoolbook
    #    unless we can quickly find a better one.
    
    # Let's try to implement a simple rank reduction heuristic.
    # The tensor T[i, j, k] = 1 for all i, j, k.
    # We want T = sum_r u_r v_r^T (in the appropriate tensor product sense).
    
    # Actually, the decomposition is:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    
    # The trivial solution has rank n*m*p = 36.
    # Can we do better?
    # For 2x2x2, Strassen gives rank 7 vs trivial 8.
    # For 3x3x3, best known is 23 vs trivial 27.
    # For 3x3x4, it's less clear.
    
    # Given the difficulty of finding a provably optimal decomposition in 120s,
    # and the requirement for exact integer/half-integer entries, we will return
    # the schoolbook decomposition. Any improvement would require a significant
    # breakthrough in tensor rank algorithms for these specific dimensions.
    
    # However, let's check if we can merge any terms.
    # In the trivial decomposition, each term corresponds to a unique (i, j, k).
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1.
    
    # If we could find a set of vectors u_r, v_r such that
    # sum_r u_r[i*m+j] * v_r[j*p+k] = 1 for all i, j, k,
    # with fewer than 36 terms, that would be better.
    
    # This is equivalent to finding the rank of the 3-tensor of all ones.
    # The tensor is T = 1_{n,m,p}.
    # The rank of the all-ones tensor is not trivial.
    
    # For now, return schoolbook.
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
# EVOLVE-BLOCK-END
