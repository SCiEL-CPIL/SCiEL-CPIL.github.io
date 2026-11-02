"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # U_s[i*m+j], V_s[j*p+k], W_s[k*n+i] for one block
    # Indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B block (2x2): 00, 01, 10, 11
    # Indices for C block (2x2): 00, 01, 10, 11
    
    # U vectors (size 4)
    U_vecs = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]
    
    # V vectors (size 4, relative to block)
    V_vecs = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, -1, 1, 0], # M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # W vectors (size 4, relative to block)
    # C indices: 00, 10, 01, 11 in flattened k*n+i?
    # k is column, i is row.
    # C[0,0] -> k=0, i=0 -> idx 0
    # C[1,0] -> k=0, i=1 -> idx 1
    # C[0,1] -> k=1, i=0 -> idx 2
    # C[1,1] -> k=1, i=1 -> idx 3
    W_vecs = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 1, 0, -1], # M2: C10 - C11
        [0, 0, 1, 1], # M3: C01 + C11
        [1, 1, 0, 0], # M4: C00 + C10
        [-1, 0, 1, 0], # M5: -C00 + C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    num_blocks = p // 2 # 7 blocks
    
    for q in range(num_blocks):
        for s in range(7):
            col = q * 7 + s
            
            # Fill U
            for idx, val in enumerate(U_vecs[s]):
                U[idx][col] = val
                
            # Fill V
            # B indices for block q:
            # B[0, 2q] -> 2q
            # B[0, 2q+1] -> 2q+1
            # B[1, 2q] -> 14 + 2q
            # B[1, 2q+1] -> 14 + 2q + 1
            b_indices = [2*q, 2*q+1, 14+2*q, 14+2*q+1]
            for idx, val in enumerate(V_vecs[s]):
                V[b_indices[idx]][col] = val
                
            # Fill W
            # C indices for block q:
            # C[0, 2q] -> k=2q, i=0 -> 2*(2q) + 0 = 4q
            # C[1, 2q] -> k=2q, i=1 -> 2*(2q) + 1 = 4q+1
            # C[0, 2q+1] -> k=2q+1, i=0 -> 2*(2q+1) + 0 = 4q+2
            # C[1, 2q+1] -> k=2q+1, i=1 -> 2*(2q+1) + 1 = 4q+3
            c_indices = [4*q, 4*q+1, 4*q+2, 4*q+3]
            for idx, val in enumerate(W_vecs[s]):
                W[c_indices[idx]][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
