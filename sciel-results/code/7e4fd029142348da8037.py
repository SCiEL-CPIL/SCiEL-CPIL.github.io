"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_DIM_START = time.time()
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = 145.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for each 2x2 block of B.
    # There are 4 such blocks (since p=8, m=2, so 8/2 = 4 blocks).
    # Each block uses rank 7. Total rank = 4 * 7 = 28.
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A indices)
    # A indices: 00->0, 01->1, 10->2, 11->3
    U_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    # Strassen coefficients for V (B indices, local to block)
    V_coeffs = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]
    
    # Strassen coefficients for W (C indices, local to block)
    W_coeffs = [
        {0: 1, 3: 1},       # M1
        {1: 1, 3: -1},      # M2
        {2: 1, 3: 1},       # M3
        {0: 1, 1: 1},       # M4
        {0: -1, 2: 1},      # M5
        {3: 1},             # M6
        {0: 1}              # M7
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
                
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2
                k_local = v_local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
                
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2
                k_local = w_local_idx // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    # Deadline guard
    if time.time() - _t0 > _deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
