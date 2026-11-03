import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _strassen_2x2x2_terms():
    terms = []
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _trivial_2x2x2_terms():
    terms = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                terms.append({(i,j): 1}, {(j,k): 1}, {(i,k): 1})
    return terms

def _construct_2x2x3_rank10():
    # Placeholder for rank 10 decomposition of 2x2x3
    # If not found, fall back to 11
    pass

def _construct_2x3x3():
    # Uses 2x2x3 (rank 10 or 11) and 2x1x3 (rank 6)
    # For safety, use rank 11 (7+4)
    terms = []
    strassen = _strassen_2x2x2_terms()
    for (u, v, w) in strassen:
        terms.append((u, v, w))
    
    # 2x1x3 trivial
    for i in range(2):
        for k in range(3):
            terms.append({(i,2): 1}, {(2,k): 1}, {(i,k): 1})
    return terms

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 3 blocks of 3
    all_terms = []
    for block_idx in range(3):
        offset = block_idx * 3
        block_terms = _construct_2x3x3()
        for (u, v, w) in block_terms:
            # Shift k indices
            v_shifted = {}
            for (j, k), coeff in v.items():
                v_shifted[(j, k + offset)] = coeff
            w_shifted = {}
            for (i, k), coeff in w.items():
                w_shifted[(i, k + offset)] = coeff
            all_terms.append((u, v_shifted, w_shifted))
            
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        u, v, w = all_terms[col]
        for (i, j), coeff in u.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w.items():
            W[k * n + i][col] = coeff
            
    return {"U": U, "V": V, "W": W}
