import numpy as np

def construct():
    n, m, p = 2, 3, 6
    # A: n x m = 2x3, B: m x p = 3x6, C: n x p = 2x6
    # U: (n*m) x rank = 6 x rank
    # V: (m*p) x rank = 18 x rank
    # W: (p*n) x rank = 12 x rank
    # Convention: W[k*n+i, r]
    
    # Try to find a low-rank decomposition
    # Naive rank is n*m*p = 36
    
    # Implement the known rank-33 solution
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = []
    
    # For a 2x3x2 block (from the current solution)
    # A (2x3): indices (i,j) -> i*3+j, i in 0..1, j in 0..2
    # B (3x2): indices (j,k) -> j*2+k, j in 0..2, k in 0..1  
    # C (2x2): indices (i,k) -> i*2+k, i in 0..1, k in 0..1
    
    # Strassen for the 2x2 x 2x2 part (j=0,1)
    # Plus 4 scalar products for the j=2 column/row
    
    # M1 = (A00+A11)(B00+B11)
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    # M2 = (A10+A11)B00
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    # M3 = A00(B01-B11)
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    # M4 = A11(B10-B00)
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    # M5 = (A00+A01)B11
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    # M6 = (A10-A00)(B00+B01)
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    # M7 = (A01-A11)(B10+B11)
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    # K1 = A02 * B20
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    # K2 = A02 * B21
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    # K3 = A12 * B20
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    # K4 = A12 * B21
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
