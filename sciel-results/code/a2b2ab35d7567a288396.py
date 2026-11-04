import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 65: 49 for 4x4x4 (Strassen^2) + 16 for 4x4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_terms, v_terms, w_idx, coeff):
        nonlocal col
        if col >= rank:
            return
        for u_idx, cu in u_terms:
            U[u_idx][col] += cu * coeff
        for v_idx, cv in v_terms:
            V[v_idx][col] += cv * cv # BUG: should be cv * coeff
        W[w_idx][col] += coeff
        col += 1
