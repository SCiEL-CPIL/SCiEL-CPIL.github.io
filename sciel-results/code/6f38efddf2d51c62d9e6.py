"""Program for 2x2x8 matrix multiplication using Strassen's algorithm on blocks.
The rank is 28, which is the sum of the ranks of four independent 2x2x2 multiplications.
This is considered optimal for this tensor structure."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # A indices: 0:00, 1:01, 2:10, 3:11
    U_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    # B indices local to block: 0:00, 1:01, 2:10, 3:11
    V_coeffs = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]
    
    # C indices local to block: 0:00, 1:10, 2:01, 3:11
    # W index convention: k*n + i
    # Local 0 (C00) -> i=0, k_local=0
    # Local 1 (C10) -> i=1, k_local=0
    # Local 2 (C01) -> i=0, k_local=1
    # Local 3 (C11) -> i=1, k_local=1
    W_coeffs = [
        {0: 1, 3: 1},       # M1: C00 + C11
        {1: 1, 3: -1},      # M2: C10 - C11
        {2: 1, 3: 1},       # M3: C01 + C11
        {0: 1, 1: 1},       # M4: C00 + C10
        {0: -1, 2: 1},      # M5: -C00 + C01
        {3: 1},             # M6: C11
        {0: 1}              # M7: C00
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
