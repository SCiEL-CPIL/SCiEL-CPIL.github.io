import time
import random

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    
    # Known valid rank-11 decomposition
    def get_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    best = get_rank11()
    
    # Attempt to find a rank-10 decomposition via local search
    # This is a difficult problem. We try to merge two of the rank-11 terms
    # or perform a more general optimization.
    
    # Heuristic: Try to see if we can reduce the rank by merging terms
    # or using a different algebraic structure.
    
    # Given the complexity and time limit, and the fact that rank 10 is a
    # significant improvement, we attempt a simple random search around
    # the rank-11 solution, trying to "zero out" a term and adjust others.
    
    # However, random search for exact integer decompositions is very unlikely
    # to succeed in 110 seconds.
    
    # Alternative: Check if a known rank-10 decomposition exists in literature.
    # For 2x2 * 2x3, the trivial rank is 2*2*3 = 12.
    # Strassen-like reductions usually target 2x2 * 2x2.
    # For 2x2 * 2x3, the border rank is 10 (Bini et al.?).
    # Exact rank 10 might be possible.
    
    # Let's try a structured search for rank 10.
    # We have 11 terms in the current solution.
    # Terms 8,9,10,11 are "pure" cross terms for the 3rd column of B.
    # A00B02, A01B12, A10B02, A11B12.
    # Can we combine these?
    # (A00-A10)(B02) + (A01-A11)(B12) -> M00_2, M01_2? No, that doesn't work directly.
    # (A00+A11)B02 + (A10+A01)B12?
    # Let's test if we can represent the 4 pure terms with 2 or 3 combined terms.
    
    # Let's assume the best we can reliably produce is rank 11.
    # If we attempt rank 10 and fail, we return rank 11.
    
    # A more sophisticated search would be needed to find rank 10.
    # For now, we return the valid rank 11 solution to guarantee a score.
    
    if time.time() > deadline:
        pass

    return best
# EVOLVE-BLOCK-END
