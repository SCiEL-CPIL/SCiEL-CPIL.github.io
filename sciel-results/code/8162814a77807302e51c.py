import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank is 49 (7 groups * 7 Strassen products)
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (U)
    # s: 0..6
    # U indices: i*m + j -> 2i + j
    # 0: 00, 1: 01, 2: 10, 3: 11
    
    U_coeffs = [
        {0: 1, 3: 1},          # s=0: a00 + a11
        {2: 1, 3: 1},          # s=1: a10 + a11
        {0: 1},                # s=2: a00
        {3: 1},                # s=3: a11
        {0: 1, 2: 1},          # s=4: a00 + a10
        {0: -1, 2: 1},         # s=5: a10 - a00
        {1: -1, 3: 1}          # s=6: a11 - a01
    ]
    
    # Strassen coefficients for B (V)
    # V indices: j*p + k -> 14j + k
    # k is local 0 or 1, mapped to 2t and 2t+1
    # j is 0 or 1
    
    # s: 0..6
    # Returns list of (j, local_k, coeff)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # s=0: b00 + b11
        [(0, 0, 1)],              # s=1: b00
        [(1, 0, 1), (1, 1, -1)],  # s=2: b10 - b11
        [(0, 1, 1), (0, 0, -1)],  # s=3: b01 - b00
        [(1, 1, 1)],              # s=4: b11
        [(0, 0, 1), (1, 0, 1)],   # s=5: b00 + b10
        [(0, 1, 1), (1, 1, 1)]    # s=6: b01 + b11
    ]
    
    # Strassen coefficients for C (W)
    # W indices: k*n + i -> 2k + i
    # local k is 0 or 1, mapped to 2t and 2t+1
    # i is 0 or 1
    
    # s: 0..6
    # Returns list of (local_k, i, coeff)
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # s=0: z00, z11
        [(1, 0, 1), (1, 1, -1)],  # s=1: z10, z11(-)
        [(0, 1, 1), (1, 1, 1)],   # s=2: z01, z11
        [(0, 0, 1), (1, 0, 1)],   # s=3: z00, z10
        [(0, 0, -1), (0, 1, 1)],  # s=4: z00(-), z01
        [(1, 1, 1)],              # s=5: z11
        [(0, 0, 1)]               # s=6: z00
    ]
    
    for t in range(7):
        for s in range(7):
            r = 7 * t + s
            
            # Fill U
            for idx, val in U_coeffs[s].items():
                U[idx][r] = val
                
            # Fill V
            for j, lk, val in V_coeffs[s]:
                k = 2 * t + lk
                v_idx = j * p + k
                V[v_idx][r] = val
                
            # Fill W
            for lk, i, val in W_coeffs[s]:
                k = 2 * t + lk
                w_idx = k * n + i
                W[w_idx][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
