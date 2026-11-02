import numpy as np
import time

DIMENSIONS = [3, 3, 3]
DEADLINE = 100  # seconds, safety margin before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank-26 decomposition terms
    # U: 9x26, V: 9x26, W: 9x26
    # Indices:
    # A: i*3+j
    # B: j*3+k
    # C: k*3+i (W index)
    
    terms = []
    
    # 1. Strassen M1
    terms.append({
        'u': [(0, 1), (4, 1)],
        'v': [(0, 1), (4, 1)],
        'w': [(0, 1), (4, 1)]
    })
    
    # 2. Strassen M2
    terms.append({
        'u': [(3, 1), (4, 1)],
        'v': [(0, 1)],
        'w': [(1, 1), (4, -1)]
    })
    
    # 3. Strassen M3
    terms.append({
        'u': [(0, 1)],
        'v': [(1, 1), (4, -1)],
        'w': [(3, 1), (4, 1)]
    })
    
    # 4. Strassen M4
    terms.append({
        'u': [(4, 1)],
        'v': [(3, 1), (0, -1)],
        'w': [(0, 1), (1, 1)]
    })
    
    # 5. Strassen M5
    terms.append({
        'u': [(0, 1), (1, 1)],
        'v': [(4, 1)],
        'w': [(0, -1), (3, 1)]
    })
    
    # 6. Strassen M6
    terms.append({
        'u': [(3, 1), (0, -1)],
        'v': [(0, 1), (1, 1)],
        'w': [(4, 1)]
    })
    
    # 7. Strassen M7
    terms.append({
        'u': [(1, 1), (4, -1)],
        'v': [(3, 1), (4, 1)],
        'w': [(0, 1)]
    })
    
    # 8-11. A_TR * B_BL
    terms.append({'u': [(2, 1)], 'v': [(6, 1)], 'w': [(0, 1)]})
    terms.append({'u': [(2, 1)], 'v': [(7, 1)], 'w': [(3, 1)]})
    terms.append({'u': [(5, 1)], 'v': [(6, 1)], 'w': [(1, 1)]})
    terms.append({'u': [(5, 1)], 'v': [(7, 1)], 'w': [(4, 1)]})
    
    # 12-15. A_TL * B_TR
    terms.append({'u': [(0, 1)], 'v': [(2, 1)], 'w': [(6, 1)]})
    terms.append({'u': [(1, 1)], 'v': [(5, 1)], 'w': [(6, 1)]})
    terms.append({'u': [(3, 1)], 'v': [(2, 1)], 'w': [(7, 1)]})
    terms.append({'u': [(4, 1)], 'v': [(5, 1)], 'w': [(7, 1)]})
    
    # 16-17. A_TR * B_BR
    terms.append({'u': [(2, 1)], 'v': [(8, 1)], 'w': [(6, 1)]})
    terms.append({'u': [(5, 1)], 'v': [(8, 1)], 'w': [(7, 1)]})
    
    # 18-21. A_BL * B_TL
    terms.append({'u': [(6, 1)], 'v': [(0, 1)], 'w': [(2, 1)]})
    terms.append({'u': [(7, 1)], 'v': [(3, 1)], 'w': [(2, 1)]})
    terms.append({'u': [(6, 1)], 'v': [(1, 1)], 'w': [(5, 1)]})
    terms.append({'u': [(7, 1)], 'v': [(4, 1)], 'w': [(5, 1)]})
    
    # 22-23. A_BR * B_BL
    terms.append({'u': [(8, 1)], 'v': [(6, 1)], 'w': [(2, 1)]})
    terms.append({'u': [(8, 1)], 'v': [(7, 1)], 'w': [(5, 1)]})
    
    # 24-25. A_BL * B_TR
    terms.append({'u': [(6, 1)], 'v': [(2, 1)], 'w': [(8, 1)]})
    terms.append({'u': [(7, 1)], 'v': [(5, 1)], 'w': [(8, 1)]})
    
    # 26. A_BR * B_BR
    terms.append({'u': [(8, 1)], 'v': [(8, 1)], 'w': [(8, 1)]})
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        for idx, coeff in term['u']:
            U[idx][r] += coeff
        for idx, coeff in term['v']:
            V[idx][r] += coeff
        for idx, coeff in term['w']:
            W[idx][r] += coeff
            
    return {"U": U, "V": V, "W": W}
