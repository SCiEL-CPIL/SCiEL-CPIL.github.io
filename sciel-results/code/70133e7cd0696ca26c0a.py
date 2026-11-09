import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # alpha[i, j, r], beta[j, k, r], gamma[k, i, r]
    # Using 0-indexed local indices for i, j, k in {0, 1}
    
    # r=0: M1
    # A: a+d -> (0,0):1, (1,1):1
    # B: e+h -> (0,0):1, (1,1):1
    # C: C00=1, C11=1 -> (0,0):1, (1,1):1
    # r=1: M2
    # A: c+d -> (1,0):1, (1,1):1
    # B: e -> (0,0):1
    # C: C10=1 -> (0,1):1
    # r=2: M3
    # A: a -> (0,0):1
    # B: g-h -> (1,0):1, (1,1):-1
    # C: C01=1 -> (1,0):1
    # r=3: M4
    # A: d -> (1,1):1
    # B: f-e -> (0,1):1, (0,0):-1
    # C: C00=1, C10=1 -> (0,0):1, (0,1):1
    # r=4: M5
    # A: a+b -> (0,0):1, (0,1):1
    # B: h -> (1,1):1
    # C: C00=-1, C01=1 -> (0,0):-1, (1,0):1
    # r=5: M6
    # A: c-a -> (1,0):1, (0,0):-1
    # B: e+f -> (0,0):1, (0,1):1
    # C: C11=1 -> (1,1):1
    # r=6: M7
    # A: b-d -> (0,1):1, (1,1):-1
    # B: g+h -> (1,0):1, (1,1):1
    # C: C00=1, C11=-1 -> (0,0):1, (1,1):-1

    # List of (alpha_dict, beta_dict, gamma_dict)
    # alpha_dict: {(i,j): coeff}
    # beta_dict: {(j,k): coeff}
    # gamma_dict: {(k,i): coeff}
    
    strassen_terms = [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1}
        ),
        (
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(1,0): 1}
        ),
        (
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (1,0): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): -1}
        )
    ]
    
    # Blocks: (i_offset, j_offset_A, j_offset_B, k_offset)
    # i_offset: 0
    # j_offset_A: 0 or 2
    # j_offset_B: 0 or 2
    # k_offset: 0 or 2
    
    blocks = [
        (0, 0, 0, 0), # A1 B11
        (0, 2, 2, 0), # A2 B21
        (0, 0, 0, 2), # A1 B12
        (0, 2, 2, 2)  # A2 B22
    ]
    
    r_base = 0
    for b_idx in range(4):
        i_off, jA_off, jB_off, k_off = blocks[b_idx]
        for t_idx in range(7):
            r = r_base + t_idx
            alpha, beta, gamma = strassen_terms[t_idx]
            
            for (li, lj), coeff_a in alpha.items():
                gi = i_off + li
                gj = jA_off + lj
                U[gi * m + gj][r] = coeff_a
            
            for (lj, lk), coeff_b in beta.items():
                gj = jB_off + lj
                gk = k_off + lk
                V[gj * p + gk][r] = coeff_b
            
            for (lk, li), coeff_c in gamma.items():
                gk = k_off + lk
                gi = i_off + li
                W[gk * n + gi][r] = coeff_c
        
        r_base += 7

    return {"U": U, "V": V, "W": W}
