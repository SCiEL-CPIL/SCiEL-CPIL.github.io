"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-structured Strassen decomposition for 6x6 matrix multiplication.
    Rank 189.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    bs = 2
    nb = 3
    total_rank = 27 * 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    output_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for t in range(7):
                    for r in range(bs):
                        for c in range(bs):
                            a_idx = r * bs + c
                            a_coeff = strassen_A[t][a_idx]
                            if a_coeff != 0:
                                U[(i * bs + r) * m + (k * bs + c)][col] = a_coeff
                            
                            b_idx = r * bs + c
                            b_coeff = strassen_B[t][b_idx]
                            if b_coeff != 0:
                                V[(k * bs + r) * p + (j * bs + c)][col] = b_coeff
                            
                            c_local_idx = r * bs + c
                            c_coeff = output_coeffs[t][c_local_idx]
                            if c_coeff != 0:
                                W[(j * bs + c) * n + (i * bs + r)][col] = c_coeff
                            
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
