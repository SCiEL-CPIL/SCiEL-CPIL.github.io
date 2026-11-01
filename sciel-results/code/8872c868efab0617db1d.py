import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    R = 11
    
    # Initialize matrices
    # U: (n*m) x R
    # V: (m*p) x R
    # W: (p*n) x R
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Define the 11 rank-1 components
    # Each component is a tuple (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # Or simply list of non-zero entries
    components = [
        # 1
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (4, 1)], 'W': [(0, 1), (3, 1)]},
        # 2
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        # 3
        {'U': [(0, 1)], 'V': [(1, 1), (4, -1)], 'W': [(1, 1), (3, 1)]},
        # 4
        {'U': [(3, 1)], 'V': [(3, 1), (0, -1)], 'W': [(0, 1), (2, 1)]},
        # 5
        {'U': [(0, 1), (1, 1)], 'V': [(4, 1)], 'W': [(0, -1), (2, 1)]},
        # 6
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # 7
        {'U': [(1, 1), (3, -1)], 'V': [(3, 1), (4, 1)], 'W': [(0, 1)]},
        # 8
        {'U': [(0, 1)], 'V': [(2, 1)], 'W': [(4, 1)]},
        # 9
        {'U': [(1, 1)], 'V': [(5, 1)], 'W': [(4, 1)]},
        # 10
        {'U': [(2, 1)], 'V': [(2, 1)], 'W': [(5, 1)]},
        # 11
        {'U': [(3, 1)], 'V': [(5, 1)], 'W': [(5, 1)]},
    ]
    
    for idx, comp in enumerate(components):
        for pos, val in comp['U']:
            U[pos][idx] = val
        for pos, val in comp['V']:
            V[pos][idx] = val
        for pos, val in comp['W']:
            W[pos][idx] = val
            
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
