"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Define the 7 Strassen terms for 2x2x2
    # Each term is a tuple of three dictionaries: U, V, W entries
    # Keys are (i,j), (j,k), (k,i) respectively
    # Values are coefficients
    
    # We define the terms for the "left" block (k=0,1) and "right" block (k=2,3)
    # The directive asks for a "middle product" style, but mathematically for 2x2x4
    # with rank 14, it decomposes into two 2x2x2 instances.
    # We implement this cleanly.
    
    # Strassen terms:
    # 1: (a00+a11)(b00+b11) -> c00, c11
    # 2: (a10+a11)b00 -> c10, c11
    # 3: a00(b01-b11) -> c01, c11
    # 4: a11(b10-b00) -> c00, c10
    # 5: (a00+a01)b11 -> c01, c00
    # 6: (a10-a00)(b00+b01) -> c11
    # 7: (a01-a11)(b10+b11) -> c00
    
    # To allow "shared support" or just clean structure, we define the coefficients
    # for U, V, W for each of the 7 terms.
    
    # U coefficients (i, j)
    # V coefficients (j, k_local) where k_local is 0 or 1
    # W coefficients (k_local, i)
    
    terms = [
        # 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # 3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # 4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # 5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,0):1, (0,0):-1}),
        # 6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # 7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Fill U, V, W for the first block (k=0,1) -> columns 0-6
    for r in range(7):
        u_dict, v_dict, w_dict = terms[r]
        
        for (i, j), val in u_dict.items():
            U[i*m + j][r] += val
            
        for (j, k), val in v_dict.items():
            # k is local index 0 or 1
            V[j*p + k][r] += val
            
        for (k, i), val in w_dict.items():
            # k is local index 0 or 1
            W[k*n + i][r] += val
            
    # Fill U, V, W for the second block (k=2,3) -> columns 7-13
    for r in range(7):
        u_dict, v_dict, w_dict = terms[r]
        col = 7 + r
        
        for (i, j), val in u_dict.items():
            U[i*m + j][col] += val
            
        for (j, k), val in v_dict.items():
            # k is local index 0 or 1, map to global k = k + 2
            k_global = k + 2
            V[j*p + k_global][col] += val
            
        for (k, i), val in w_dict.items():
            # k is local index 0 or 1, map to global k = k + 2
            k_global = k + 2
            W[k_global*n + i][col] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
