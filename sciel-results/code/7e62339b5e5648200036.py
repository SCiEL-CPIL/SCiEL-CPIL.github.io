"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Define the rank-7 Strassen decomposition for 2x2x2 multiplication.
    # Indices:
    # U: A[i, j] -> row i*m + j (0..3)
    # V: B[j, k_local] -> row j*2 + k_local (0..3) in local block
    # W: C[i, k_local] -> row k_local*n + i (0..3) in local block (Trace convention: C[k, i] in standard? 
    # Task says: W[k*n+i, r]. So W row index is k*n + i.
    # In 2x2x2, k is 0..1, i is 0..1. Rows 0..3.
    
    # Strassen factors: (U_factors, V_factors, W_factors)
    # Each factor is a list of (index, coeff)
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    s_factors = [
        # M1
        ([ (0*2+0, 1), (1*2+1, 1) ],  # a00, a11
         [ (0*2+0, 1), (1*2+1, 1) ],  # b00, b11
         [ (0*2+0, 1), (1*2+1, 1) ]), # c00, c11
        # M2
        ([ (1*2+0, 1), (1*2+1, 1) ],  # a10, a11
         [ (0*2+0, 1) ],              # b00
         [ (1*2+0, 1), (1*2+1, -1) ]),# c10, c11
        # M3
        ([ (0*2+0, 1) ],              # a00
         [ (0*2+1, 1), (1*2+1, -1) ],# b01, b11
         [ (0*2+1, 1), (1*2+1, 1) ]),# c01, c11
        # M4
        ([ (1*2+1, 1) ],              # a11
         [ (1*2+0, 1), (0*2+0, -1) ],# b10, b00
         [ (0*2+0, 1), (1*2+0, 1) ]),# c00, c10
        # M5
        ([ (0*2+0, 1), (0*2+1, 1) ],  # a00, a01
         [ (1*2+1, 1) ],              # b11
         [ (0*2+0, -1), (0*2+1, 1) ]),# c00, c01
        # M6
        ([ (1*2+0, 1), (0*2+0, -1) ], # a10, a00
         [ (0*2+0, 1), (0*2+1, 1) ],  # b00, b01
         [ (1*2+1, 1) ]),             # c11
        # M7
        ([ (0*2+1, 1), (1*2+1, -1) ], # a01, a11
         [ (1*2+0, 1), (1*2+1, 1) ],  # b10, b11
         [ (0*2+0, 1) ]),             # c00
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_entries, v_entries, w_entries = s_factors[s]
            
            for (idx, coeff) in u_entries:
                U[idx][col] = coeff
            
            for (idx, coeff) in v_entries:
                # Map local V index (j*2+k) to global V index (j*p + (2*t+k))
                j = idx // 2
                k = idx % 2
                global_v_idx = j * p + (2 * t + k)
                V[global_v_idx][col] = coeff
            
            for (idx, coeff) in w_entries:
                # Map local W index (k*2+i) to global W index ((2*t+k)*n + i)
                k = idx // 2
                i = idx % 2
                global_w_idx = (2 * t + k) * n + i
                W[global_w_idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
