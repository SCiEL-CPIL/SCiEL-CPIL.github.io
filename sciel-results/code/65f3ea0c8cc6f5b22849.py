"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Rank 21 decomposition using Strassen's algorithm for 3 blocks
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (U matrix)
    # Indices: 0: A00, 1: A01, 2: A10, 3: A11
    # M1..M7 correspond to k=0..6
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # Strassen coefficients for B (V matrix)
    # Local indices: j in {0,1}, k_local in {0,1}
    # Global k = 2*r + k_local
    # V index = j * p + global_k
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    # Strassen coefficients for C (W matrix)
    # W index = k_global * n + i
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: C00 + C11
        [(1, 0, 1), (1, 1, -1)],     # M2: C10 - C11
        [(0, 1, 1), (1, 1, 1)],      # M3: C01 + C11
        [(0, 0, 1), (1, 0, 1)],      # M4: C00 + C10
        [(0, 0, -1), (0, 1, 1)],     # M5: -C00 + C01
        [(1, 1, 1)],                 # M6: C11
        [(0, 0, 1)]                  # M7: C00
    ]
    
    # Process 3 blocks of 2x2
    for b in range(3):
        for t in range(7):
            r = b * 7 + t
            
            # Fill U
            for idx, val in U_coeffs[t].items():
                U[idx][r] = val
                
            # Fill V
            for (j, k_local, val) in V_coeffs[t]:
                k_global = 2 * b + k_local
                v_idx = j * p + k_global
                V[v_idx][r] = val
                
            # Fill W
            for (i, k_local, val) in W_coeffs[t]:
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
