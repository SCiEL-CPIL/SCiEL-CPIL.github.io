import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    The tensor is a direct sum of three 2x2x2 multiplication tensors.
    Rank of 2x2x2 is 7. Total rank is 3 * 7 = 21.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the target tensor for verification (optional but good for sanity)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def strassen_21():
        """Build rank-21 Strassen decomposition."""
        rank = 21
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Strassen steps for 2x2x2
        # Indices: 0:00, 1:01, 2:10, 3:11
        strassen_steps = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
        ]
        
        def map_a(local_idx):
            return local_idx
            
        def map_b(local_idx, g):
            k0 = 2 * g
            k1 = 2 * g + 1
            if local_idx == 0: return 0 * 6 + k0
            elif local_idx == 1: return 0 * 6 + k1
            elif local_idx == 2: return 1 * 6 + k0
            else: return 1 * 6 + k1

        def map_c(local_idx, g):
            k0 = 2 * g
            k1 = 2 * g + 1
            if local_idx == 0: return k0 * 2 + 0
            elif local_idx == 1: return k1 * 2 + 0
            elif local_idx == 2: return k0 * 2 + 1
            else: return k1 * 2 + 1

        for g in range(3):
            for s in range(7):
                r = 7 * g + s
                step = strassen_steps[s]
                for local_idx in range(4):
                    u_val = step['u'][local_idx]
                    if u_val != 0:
                        U[map_a(local_idx), r] = u_val
                    v_val = step['v'][local_idx]
                    if v_val != 0:
                        V[map_b(local_idx, g), r] = v_val
                    w_val = step['w'][local_idx]
                    if w_val != 0:
                        W[map_c(local_idx, g), r] = w_val
        return U, V, W
    
    U, V, W = strassen_21()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
