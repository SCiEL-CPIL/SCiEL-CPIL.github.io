import numpy as np

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2
    # u, v, w are length 4
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    for t in range(3):
        for s in range(7):
            c = t * 7 + s
            u_local, v_local, w_local = terms[s]
            
            # U is same for all blocks
            U[:, c] = u_local
            
            # V mapping: local j in {0,1}, k_local in {0,1} -> global j, k_global = 2*t + k_local
            for j in range(2):
                for k_local in range(2):
                    v_idx_local = j * 2 + k_local
                    v_idx_global = j * 6 + (2 * t + k_local)
                    V[v_idx_global, c] = v_local[v_idx_local]
                    
            # W mapping: local k_local in {0,1}, i in {0,1} -> global k_global = 2*t + k_local, i
            for k_local in range(2):
                for i in range(2):
                    w_idx_local = k_local * 2 + i
                    w_idx_global = (2 * t + k_local) * 2 + i
                    W[w_idx_global, c] = w_local[w_idx_local]
                    
    return {"U": U, "V": V, "W": W}
