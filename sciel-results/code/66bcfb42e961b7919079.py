"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 7 = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # s: 0..6
    # A coeffs: dict index (i, j) -> coeff
    # B coeffs: dict index (j, k) -> coeff
    # C coeffs: dict index (i, k) -> coeff
    
    strassen_A = [
        {0: 1, 3: 1},       # M0: a00 + a11
        {2: 1, 3: 1},       # M1: a10 + a11
        {0: 1},             # M2: a00
        {3: 1},             # M3: a11
        {0: 1, 1: 1},       # M4: a00 + a01
        {2: 1, 0: -1},      # M5: a10 - a00
        {1: 1, 3: -1}       # M6: a01 - a11
    ]
    
    strassen_B = [
        {0: 1, 3: 1},       # M0: b00 + b11
        {0: 1},             # M1: b00
        {1: 1, 3: -1},      # M2: b01 - b11
        {2: 1, 0: -1},      # M3: b10 - b00
        {3: 1},             # M4: b11
        {0: 1, 1: 1},       # M5: b00 + b01
        {2: 1, 3: 1}        # M6: b10 + b11
    ]
    
    strassen_C = [
        {0: 1, 3: 1},       # M0: c00 + c11
        {2: 1, 3: -1},      # M1: c10 - c11
        {1: 1, 3: 1},       # M2: c01 + c11
        {0: 1, 2: 1},       # M3: c00 + c10
        {0: -1, 1: 1},      # M4: -c00 + c01
        {3: 1},             # M5: c11
        {0: 1}              # M6: c00
    ]
    
    for block in range(7):
        for s in range(7):
            r = block * 7 + s
            
            # Fill U
            for idx, coeff in strassen_A[s].items():
                U[idx][r] = coeff
                
            # Fill V
            # V index: j * p + k
            # block corresponds to columns 2*block and 2*block+1
            # B indices in local block:
            # 0 -> (0, 0) -> global (0, 2*block) -> V[0*p + 2*block]
            # 1 -> (0, 1) -> global (0, 2*block+1) -> V[0*p + 2*block+1]
            # 2 -> (1, 0) -> global (1, 2*block) -> V[1*p + 2*block]
            # 3 -> (1, 1) -> global (1, 2*block+1) -> V[1*p + 2*block+1]
            for idx, coeff in strassen_B[s].items():
                local_j = idx // 2
                local_k = idx % 2
                global_j = local_j
                global_k = 2 * block + local_k
                v_idx = global_j * p + global_k
                V[v_idx][r] = coeff
                
            # Fill W
            # W index: k * n + i
            # C indices in local block:
            # 0 -> (0, 0) -> global (0, 2*block) -> W[2*block*n + 0]
            # 1 -> (0, 1) -> global (0, 2*block+1) -> W[(2*block+1)*n + 0]
            # 2 -> (1, 0) -> global (1, 2*block) -> W[2*block*n + 1]
            # 3 -> (1, 1) -> global (1, 2*block+1) -> W[(2*block+1)*n + 1]
            for idx, coeff in strassen_C[s].items():
                local_i = idx // 2
                local_k = idx % 2
                global_i = local_i
                global_k = 2 * block + local_k
                w_idx = global_k * n + global_i
                W[w_idx][r] = coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
