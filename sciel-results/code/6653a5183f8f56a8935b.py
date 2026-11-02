"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U(i,j), V(j,k), W(k,i)
    
    # U coefficients (A_ij)
    # M1: A00+A11
    # M2: A10+A11
    # M3: A00
    # M4: A11
    # M5: A00+A01
    # M6: A10-A00
    # M7: A01-A11
    
    U_strassen = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 0, 1, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [-1, 0, 1, 0],# M6: A10, -A00
        [0, 1, 0, -1] # M7: A01, -A11
    ]
    
    # V coefficients (B_jk)
    # M1: B00+B11
    # M2: B00
    # M3: B01-B11
    # M4: B10-B00
    # M5: B11
    # M6: B00+B01
    # M7: B10+B11
    
    V_strassen = [
        [1, 0, 0, 1], # M1: B00, B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01, -B11
        [0, 0, 1, -1],# M4: B10, -B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00, B01
        [0, 0, 1, 1]  # M7: B10, B11
    ]
    
    # W coefficients (C_ki)
    # M1: C00+C11
    # M2: C10-C11
    # M3: C01+C11
    # M4: C00+C10
    # M5: -C00+C01
    # M6: C11
    # M7: C00
    
    W_strassen = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, -1],# M2: C10, -C11
        [0, 0, 1, 1], # M3: C01, C11
        [1, 1, 0, 0], # M4: C00, C10
        [-1, 0, 1, 0],# M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for q in range(num_blocks):
        for s in range(7):
            r = q * 7 + s
            
            # Fill U
            for idx in range(4):
                if U_strassen[s][idx] != 0:
                    U[idx][r] = U_strassen[s][idx]
            
            # Fill V
            # Local B index 0 (j=0,k=0) -> global j=0, k=2*q
            # Local B index 1 (j=0,k=1) -> global j=0, k=2*q+1
            # Local B index 2 (j=1,k=0) -> global j=1, k=2*q
            # Local B index 3 (j=1,k=1) -> global j=1, k=2*q+1
            for idx in range(4):
                if V_strassen[s][idx] != 0:
                    j = idx // 2
                    k_local = idx % 2
                    k = 2 * q + k_local
                    v_idx = j * p + k
                    V[v_idx][r] = V_strassen[s][idx]
            
            # Fill W
            # Local C index 0 (k=0,i=0) -> global k=2*q, i=0
            # Local C index 1 (k=0,i=1) -> global k=2*q, i=1
            # Local C index 2 (k=1,i=0) -> global k=2*q+1, i=0
            # Local C index 3 (k=1,i=1) -> global k=2*q+1, i=1
            for idx in range(4):
                if W_strassen[s][idx] != 0:
                    k_local = idx // 2
                    i = idx % 2
                    k = 2 * q + k_local
                    w_idx = k * n + i
                    W[w_idx][r] = W_strassen[s][idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
