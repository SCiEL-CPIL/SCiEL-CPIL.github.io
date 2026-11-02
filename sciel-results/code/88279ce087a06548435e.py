"""Rewritten program for 2x2x8 matrix multiplication using Strassen's algorithm on blocks."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    U_coeffs = [
        {0: 1.0, 3: 1.0},       # M1
        {2: 1.0, 3: 1.0},       # M2
        {0: 1.0},                # M3
        {3: 1.0},                # M4
        {0: 1.0, 1: 1.0},       # M5
        {2: 1.0, 0: -1.0},      # M6
        {1: 1.0, 3: -1.0}       # M7
    ]
    V_coeffs = [
        {0: 1.0, 3: 1.0},       # M1
        {0: 1.0},                # M2
        {1: 1.0, 3: -1.0},      # M3
        {2: 1.0, 0: -1.0},      # M4
        {3: 1.0},                # M5
        {0: 1.0, 1: 1.0},       # M6
        {2: 1.0, 3: 1.0}        # M7
    ]
    W_coeffs = [
        {0: 1.0, 3: 1.0},       # M1
        {1: 1.0, 3: -1.0},      # M2
        {2: 1.0, 3: 1.0},       # M3
        {0: 1.0, 1: 1.0},       # M4
        {0: -1.0, 2: 1.0},      # M5
        {3: 1.0},                # M6
        {0: 1.0}                 # M7
    ]
    num_blocks = p // 2
    for q in range(num_blocks):
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
