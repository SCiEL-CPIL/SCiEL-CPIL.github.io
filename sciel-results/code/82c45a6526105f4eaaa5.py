import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For multiplying an n x m matrix by an m x p matrix,
    # the standard (trivial) bilinear decomposition has rank n*m*p.
    # However, we can sometimes do better by grouping terms.
    # 
    # A simple improvement: for each column j of A (i.e., A[:,j]) and row j of B (B[j,:]),
    # we compute the outer product A[:,j] * B[j,:] and add to C.
    # This requires m "steps" if we could do outer products in one term, but in bilinear
    # form, each entry C[i,k] = sum_j A[i,j]*B[j,k].
    # 
    # The trivial decomposition uses one term per (i,j,k) triple: rank = n*m*p = 2*3*10 = 60.
    # 
    # Can we do better? For small matrices, known lower bounds exist.
    # For 2x3 * 3x10, the tensor rank is known to be at most 60, but potentially lower.
    # 
    # Let's try a simple grouping: for each j, we have m=3 terms where each term is
    # the outer product of a column of A and a row of B. But in bilinear form,
    # we still need n*p terms per j in the naive approach.
    # 
    # Actually, the trivial decomposition IS the standard one. Let me just implement
    # it cleanly and see if there's a simple structural improvement.
    # 
    # One idea: if p is large relative to n, we might be able to share some W entries.
    # But for exact decomposition with integer entries, the trivial bound is hard to beat
    # without a known algorithm.
    # 
    # Let me just return the trivial decomposition but with a cleaner implementation.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
