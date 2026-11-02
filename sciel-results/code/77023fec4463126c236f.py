import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms
    # u: list of (row, col, coeff)
    # v: list of (row, col, coeff)
    # w: dict (row, col) -> coeff
    
    terms = [
        # M1
        {'u': [((0,0), 1), ((1,1), 1)], 'v': [((0,0), 1), ((1,1), 1)], 'w': {(0,0): 1, (1,1): 1}},
        # M2
        {'u': [((1,0), 1), ((1,1), 1)], 'v': [((0,0), 1)], 'w': {(1,0): 1, (1,1): -1}},
        # M3
        {'u': [((0,0), 1)], 'v': [((1,0), 1), ((1,1), -1)], 'w': {(0,1): 1, (1,1): 1}},
        # M4
        {'u': [((1,1), 1)], 'v': [((0,1), 1), ((0,0), -1)], 'w': {(0,0): 1, (1,0): 1}},
        # M5
        {'u': [((0,0), 1), ((0,1), 1)], 'v': [((1,1), 1)], 'w': {(0,0): -1, (0,1): 1}},
        # M6
        {'u': [((1,0), 1), ((0,0), -1)], 'v': [((0,0), 1), ((0,1), 1)], 'w': {(1,1): 1}},
        # M7
        {'u': [((0,1), 1), ((1,1), -1)], 'v': [((1,0), 1), ((1,1), 1)], 'w': {(0,0): 1, (1,1): 1}}
    ]
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for t in range(7):
                    for (u_r, u_c), coeff in terms[t]['u']:
                        idx_u = (6*I + u_r) * m + (6*J + u_c)
                        U[idx_u][col] += coeff
                    
                    for (v_r, v_c), coeff in terms[t]['v']:
                        idx_v = (6*J + v_r) * p + (6*K + v_c)
                        V[idx_v][col] += coeff
                    
                    for (w_r, w_c), coeff in terms[t]['w'].items():
                        idx_w = (6*K + w_c) * n + (6*I + w_r)
                        W[idx_w][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
