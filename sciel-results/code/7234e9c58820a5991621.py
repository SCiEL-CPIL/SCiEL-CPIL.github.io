"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 Strassen-based decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U vectors for the 7 Strassen terms
    # A = [[a, b], [c, d]]
    # U indices: 0:a, 1:b, 2:c, 3:d
    U_terms = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1]   # M7: b-d
    ]
    
    # V local vectors for the 7 Strassen terms
    # B block = [[e, f], [g, h]]
    # Local indices: 0:e, 1:f, 2:g, 3:h
    V_terms_local = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1]    # M7: g+h
    ]
    
    # W local vectors for the 7 Strassen terms
    # C block = [[C00, C01], [C10, C11]]
    # W index convention: W[k*n+i, r] where k is col, i is row.
    # Local C indices in W block:
    # C00 (r=0, c=0) -> idx 0
    # C10 (r=1, c=0) -> idx 1
    # C01 (r=0, c=1) -> idx 2
    # C11 (r=1, c=1) -> idx 3
    W_terms_local = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 1, 0, -1],  # M2: C10 - C11
        [0, 0, 1, 1],   # M3: C01 + C11
        [1, 1, 0, 0],   # M4: C00 + C10
        [-1, 0, 1, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ]
    
    # Construct the full matrices
    for j in range(4):  # 4 blocks of 2x2 in the 2x8 matrix
        for k in range(7):  # 7 Strassen terms per block
            r = j * 7 + k
            
            # Fill U (same for all blocks)
            for i in range(4):
                U[i][r] = U_terms[k][i]
            
            # Fill V (map local indices to global V indices)
            # B is 2x8. Block j covers columns 4j to 4j+3.
            # V index = row * 8 + col.
            # e=B[0, 4j] -> 4j
            # f=B[0, 4j+1] -> 4j+1
            # g=B[1, 4j] -> 8+4j
            # h=B[1, 4j+1] -> 9+4j
            v_map = [4*j, 4*j+1, 8+4*j, 9+4*j]
            for i in range(4):
                V[v_map[i]][r] = V_terms_local[k][i]
            
            # Fill W (map local indices to global W indices)
            # C is 2x8. Block j covers columns 4j to 4j+3.
            # W index = col * 2 + row.
            # C00 (col=4j, row=0) -> 8j
            # C10 (col=4j, row=1) -> 8j+1
            # C01 (col=4j+1, row=0) -> 8j+2
            # C11 (col=4j+1, row=1) -> 8j+3
            w_map = [8*j, 8*j+1, 8*j+2, 8*j+3]
            for i in range(4):
                W[w_map[i]][r] = W_terms_local[k][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
