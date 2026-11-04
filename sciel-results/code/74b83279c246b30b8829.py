import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = [
        [
            ('U', 0, 0, 1), ('U', 1, 1, 1),
            ('V', 0, 0, 1), ('V', 1, 1, 1),
            ('W', 0, 0, 1), ('W', 1, 1, 1)
        ],
        [
            ('U', 1, 0, 1), ('U', 1, 1, 1),
            ('V', 0, 0, 1),
            ('W', 1, 0, 1), ('W', 1, 1, -1)
        ],
        [
            ('U', 0, 0, 1),
            ('V', 0, 1, 1), ('V', 1, 1, -1),
            ('W', 0, 1, 1), ('W', 1, 1, 1)
        ],
        [
            ('U', 1, 1, 1),
            ('V', 1, 0, 1), ('V', 0, 0, -1),
            ('W', 0, 0, 1), ('W', 1, 0, 1)
        ],
        [
            ('U', 0, 0, 1), ('U', 0, 1, 1),
            ('V', 1, 1, 1),
            ('W', 0, 0, -1), ('W', 0, 1, 1)
        ],
        [
            ('U', 1, 0, 1), ('U', 0, 0, -1),
            ('V', 0, 0, 1), ('V', 0, 1, 1),
            ('W', 1, 1, 1)
        ],
        [
            ('U', 0, 1, 1), ('U', 1, 1, -1),
            ('V', 1, 0, 1), ('V', 1, 1, 1),
            ('W', 0, 0, 1), ('W', 1, 1, -1)
        ]
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 2)
    ]
    
    col_idx = 0
    for i0, j0, j1, k0 in blocks:
        if time.time() > deadline - 1.0:
            break
        for term in terms:
            if time.time() > deadline - 1.0:
                break
            for mat, r, c, val in term:
                if mat == 'U':
                    u_idx = (i0 + r) * m + (j0 + c)
                    U[u_idx][col_idx] += val
                elif mat == 'V':
                    v_idx = (j1 + r) * p + (k0 + c)
                    V[v_idx][col_idx] += val
                elif mat == 'W':
                    w_idx = (k0 + c) * n + (i0 + r)
                    W[w_idx][col_idx] += val
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
