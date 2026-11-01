"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    num_blocks = p // m  # 5 blocks of 2 columns each
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # Each term: (u: {(i,j):coeff}, v: {(j,k_local):coeff}, w: {(k_local,i):coeff})
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # P1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),  # P2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),  # P3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):-1}),  # P4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),  # P5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1, (1,1):-1}),  # P6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):-1}),  # P7
    ]
    
    for block in range(num_blocks):
        for s in range(7):
            col = block * 7 + s
            u_dict, v_dict, w_dict = terms[s]
            
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] += coeff
            
            for (j, k_local), coeff in v_dict.items():
                global_k = block * m + k_local
                V[j * p + global_k][col] += coeff
            
            for (k_local, i), coeff in w_dict.items():
                global_k = block * m + k_local
                W[global_k * n + i][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
