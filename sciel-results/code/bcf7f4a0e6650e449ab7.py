import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Uses Strassen's algorithm on three 2x2 blocks, yielding rank 21.
    This is the known minimum rank for this problem.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen steps for 2x2 matrix multiplication
    # Local indices 0..3 correspond to:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    strassen_steps = [
        # M1 = (A00+A11)(B00+B11) -> C00+C11
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        # M2 = (A10+A11)B00 -> C10
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, 0]},
        # M3 = A00(B01-B11) -> C01
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 0]},
        # M4 = A11(B10-B00) -> C00+C10
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        # M5 = (A00+A01)B11 -> C00+C01
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]}, # Wait, check signs
        # M6 = (A10-A00)(B00+B01) -> C11
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        # M7 = (A01-A11)(B10+B11) -> C00+C11
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]}, # Wait, check signs
    ]
    
    # Re-verify signs from previous code which was verified correct
    # Previous code:
    # 0: u[1,0,0,1] v[1,0,0,1] w[1,0,0,1]
    # 1: u[0,0,1,1] v[1,0,0,0] w[0,0,1,-1]
    # 2: u[1,0,0,0] v[0,1,0,-1] w[0,1,0,1]
    # 3: u[0,0,0,1] v[-1,0,1,0] w[1,0,1,0]
    # 4: u[1,1,0,0] v[0,0,0,1] w[-1,1,0,0]
    # 5: u[-1,0,1,0] v[1,1,0,0] w[0,0,0,1]
    # 6: u[0,1,0,-1] v[0,0,1,1] w[1,0,0,0]
    
    # I will use the verified steps from the previous program.
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
        if local_idx == 0:
            return 0 * 6 + k0
        elif local_idx == 1:
            return 0 * 6 + k1
        elif local_idx == 2:
            return 1 * 6 + k0
        else:
            return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return k0 * 2 + 0
        elif local_idx == 1:
            return k1 * 2 + 0
        elif local_idx == 2:
            return k0 * 2 + 1
        else:
            return k1 * 2 + 1

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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
