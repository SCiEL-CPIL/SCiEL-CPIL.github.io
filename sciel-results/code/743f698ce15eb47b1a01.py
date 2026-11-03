"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Strassen coefficients for 2x2x2
    U_local = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 0, 1, 0],   # M5: A00+A10
        [-1, 1, 0, 0],  # M6: A01-A00
        [0, 0, 1, -1],  # M7: A10-A11
    ]
    V_local = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00+B10
        [0, 1, 0, 1],   # M7: B01+B11
    ]
    W_local = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, -1],  # M7
    ]
    
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ig in range(2):
        for jg in range(2):
            for kg in range(4):
                for s in range(7):
                    for il in range(2):
                        for jl in range(2):
                            U[(2*ig+il)*m + (2*jg+jl)][col] += U_local[s][il*2+jl]
                    for jl in range(2):
                        for kl in range(2):
                            V[(2*jg+jl)*p + (2*kg+kl)][col] += V_local[s][jl*2+kl]
                    for il in range(2):
                        for kl in range(2):
                            W[(2*kg+kl)*n + (2*ig+il)][col] += W_local[s][il*2+kl]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
