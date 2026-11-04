import time
import numpy as np

TIME_LIMIT = 140

def construct():
    n, m, p = 3, 4, 6
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    terms = []
    
    def add_schoolbook(i, j, k):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        terms.append({
            'U': {u_idx: 1},
            'V': {v_idx: 1},
            'W': {w_idx: 1}
        })

    # Define the 6 blocks of size 2x2x2
    # i in 0,1; j in 0,1,2,3; k in 0,1,2,3,4,5
    # We can split j into (0,1) and (2,3)
    # We can split k into (0,1), (2,3), (4,5)
    # i is only 0,1 for the 2x2 blocks, i=2 is separate.
    
    blocks_222 = []
    for bj in [0, 2]:
        for bk in [0, 2, 4]:
            blocks_222.append((0, bj, bk)) # i0=0
            
    blocks_122 = []
    for bj in [0, 2]:
        for bk in [0, 2, 4]:
            blocks_122.append((2, bj, bk)) # i0=2

    # Strassen terms for 2x2x2
    # Local indices i', j', k' in 0..1
    # U[i', j'], V[j', k'], W[k', i']
    # T[i', j', k'] = sum U[i',j',r] V[j',k',r] W[k',i',r] = 1
    
    # Standard Strassen decomposition
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): -1}, {(1,1): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(0,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): -1, (0,1): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,1): 1, (1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (1,0): 1})
    ]
    
    # Validate Strassen terms
    def validate_strassen():
        tensor_sum = np.zeros((2,2,2))
        for u_vals, v_vals, w_vals in strassen_terms:
            for (u, v), uv in u_vals.items():
                for (x, y), vv in v_vals.items():
                    if v == x:
                        for (k_l, i_l), wv in w_vals.items():
                            tensor_sum[u, v, y] += uv * vv * wv
        return np.allclose(tensor_sum, 1.0, atol=1e-9)

    use_strassen = validate_strassen()
    
    # Add terms for 2x2x2 blocks
    for i0, j0, k0 in blocks_222:
        if use_strassen:
            for u_vals, v_vals, w_vals in strassen_terms:
                term = {'U': {}, 'V': {}, 'W': {}}
                for (u, v), val in u_vals.items():
                    i = i0 + u
                    j = j0 + v
                    term['U'][i*m + j] = val
                for (x, y), val in v_vals.items():
                    j = j0 + x
                    k = k0 + y
                    term['V'][j*p + k] = val
                for (k_l, i_l), val in w_vals.items():
                    k = k0 + k_l
                    i = i0 + i_l
                    term['W'][k*n + i] = val
                terms.append(term)
        else:
            for u in range(2):
                for v in range(2):
                    for y in range(2):
                        i = i0 + u
                        j = j0 + v
                        k = k0 + y
                        add_schoolbook(i, j, k)

    # Add terms for 1x2x2 blocks (Schoolbook only, as Strassen is for 2x2x2)
    for i0, j0, k0 in blocks_122:
        for v in range(2):
            for y in range(2):
                i = i0
                j = j0 + v
                k = k0 + y
                add_schoolbook(i, j, k)

    # Assemble U, V, W
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r, term in enumerate(terms):
        for idx, val in term['U'].items():
            U[idx][r] = val
        for idx, val in term['V'].items():
            V[idx][r] = val
        for idx, val in term['W'].items():
            W[idx][r] = val
            
    return {'U': U, 'V': V, 'W': W}
