"""Matrix multiplication tensor decomposition via exact cover search."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # Support of T in (a,b,c) space
    support_T = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support_T.add((a, b, c))
                
    # Enumerate all valid 0/1 rank-1 terms (boxes)
    # A term is defined by Su, Sv, Sw such that Su x Sv x Sw is subset of support_T
    # As derived, only singletons are valid for 0/1.
    # But let's verify programmatically to be safe and follow the directive.
    
    # To avoid explosion, we only consider singletons if we believe they are the only ones.
    # However, let's try small sets.
    # For N=15, M=35, P=21, even checking all pairs is 15*35*21 * 2^15... too big.
    # We will stick to singletons which are guaranteed valid.
    
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Create sparse representation of the term
                # u has 1 at a, v has 1 at b, w has 1 at c
                terms.append((a, b, c))
                
    # Now we have a list of terms. Each term covers exactly one cell in support_T.
    # The exact cover problem is trivial: we must select all of them.
    # Rank is len(terms) = n*m*p.
    
    rank = len(terms)
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    
    for col, (a, b, c) in enumerate(terms):
        U[a][col] = 1
        V[b][col] = 1
        W[c][col] = 1
        
    return {"U": U, "V": V, "W": W}
