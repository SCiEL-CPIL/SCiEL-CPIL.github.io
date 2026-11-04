import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_steps_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    atomic_terms = []
    
    atomic_terms.append({
        'U': [(0, 1)],
        'V': [(0, 1)],
        'W': [(0, 1)]
    })
    atomic_terms.append({
        'U': [(1, 1)],
        'V': [(3, 1)],
        'W': [(0, 1)]
    })
    atomic_terms.append({
        'U': [(2, 1)],
        'V': [(0, 1)],
        'W': [(1, 1)]
    })
    atomic_terms.append({
        'U': [(3, 1)],
        'V': [(3, 1)],
        'W': [(1, 1)]
    })
    atomic_terms.append({
        'U': [(0, 1), (3, 1)],
        'V': [(1, 1), (5, 1)],
        'W': [(2, 1), (5, 1)]
    })
    atomic_terms.append({
        'U': [(2, 1), (3, 1)],
        'V': [(1, 1)],
        'W': [(3, 1), (5, -1)]
    })
    atomic_terms.append({
        'U': [(0, 1)],
        'V': [(2, 1), (5, -1)],
        'W': [(4, 1), (5, 1)]
    })
    atomic_terms.append({
        'U': [(3, 1)],
        'V': [(4, 1), (1, -1)],
        'W': [(2, 1), (3, 1)]
    })
    atomic_terms.append({
        'U': [(0, 1), (1, 1)],
        'V': [(5, 1)],
        'W': [(2, -1), (4, 1)]
    })
    atomic_terms.append({
        'U': [(2, 1), (0, -1)],
        'V': [(1, 1), (2, 1)],
        'W': [(5, 1)]
    })
    atomic_terms.append({
        'U': [(1, 1), (3, -1)],
        'V': [(4, 1), (5, 1)],
        'W': [(2, 1)]
    })
    
    for t in range(7):
        if t == 0:
            a_blocks = [((0,0), 1), ((1,1), 1)]
            b_blocks = [((0,0), 1), ((1,1), 1)]
            c_blocks = [((0,0), 1), ((1,1), 1)]
        elif t == 1:
            a_blocks = [((1,0), 1), ((1,1), 1)]
            b_blocks = [((0,0), 1)]
            c_blocks = [((1,0), 1), ((1,1), -1)]
        elif t == 2:
            a_blocks = [((0,0), 1)]
            b_blocks = [((0,1), 1), ((1,1), -1)]
            c_blocks = [((0,1), 1), ((1,1), 1)]
        elif t == 3:
            a_blocks = [((1,1), 1)]
            b_blocks = [((1,0), 1), ((0,0), -1)]
            c_blocks = [((0,0), 1), ((1,0), 1)]
        elif t == 4:
            a_blocks = [((0,0), 1), ((0,1), 1)]
            b_blocks = [((1,1), 1)]
            c_blocks = [((0,0), -1), ((0,1), 1)]
        elif t == 5:
            a_blocks = [((1,0), 1), ((0,0), -1)]
            b_blocks = [((0,0), 1), ((0,1), 1)]
            c_blocks = [((1,1), 1)]
        elif t == 6:
            a_blocks = [((0,1), 1), ((1,1), -1)]
            b_blocks = [((1,0), 1), ((1,1), 1)]
            c_blocks = [((0,0), 1)]
        
        for s in range(11):
            col = t * 11 + s
            term = atomic_terms[s]
            
            for (u_local, val_u) in term['U']:
                i = u_local // 2
                j = u_local % 2
                for (bi, bj), coeff_a in a_blocks:
                    u_global = (bi * 2 + i) * m + (bj * 2 + j)
                    U[u_global][col] += coeff_a * val_u
            
            for (v_local, val_v) in term['V']:
                j = v_local // 3
                k = v_local % 3
                for (bi, bj), coeff_b in b_blocks:
                    v_global = (bi * 2 + j) * p + (bj * 3 + k)
                    V[v_global][col] += coeff_b * val_v
            
            for (w_local, val_w) in term['W']:
                k = w_local // 2
                i = w_local % 2
                for (bi, bj), coeff_c in c_blocks:
                    w_global = (bj * 3 + k) * n + (bi * 2 + i)
                    W[w_global][col] += coeff_c * val_w

    return {"U": U, "V": V, "W": W}
