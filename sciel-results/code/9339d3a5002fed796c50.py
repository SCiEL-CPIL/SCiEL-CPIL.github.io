import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local Strassen coefficients for 2x2 matrices
    # U_l[i] for i in 0..3
    U_loc = [
        [1, 0, 0, 1], # l=0
        [0, 0, 1, 1], # l=1
        [1, 0, 0, 0], # l=2
        [0, 0, 0, 1], # l=3
        [1, 1, 0, 0], # l=4
        [-1, 0, 1, 0],# l=5
        [0, 1, 0, -1] # l=6
    ]
    V_loc = [
        [1, 0, 0, 1], # l=0
        [1, 0, 0, 0], # l=1
        [0, 1, 0, -1],# l=2
        [-1, 0, 1, 0],# l=3
        [0, 0, 0, 1], # l=4
        [1, 1, 0, 0], # l=5
        [0, 0, 1, 1]  # l=6
    ]
    W_loc = [
        [1, 0, 0, 1], # l=0
        [0, 0, 1, -1],# l=1
        [0, 1, 0, 1], # l=2
        [1, 0, 1, 0], # l=3
        [-1, 1, 0, 0],# l=4
        [0, 0, 0, 1], # l=5
        [1, 0, 0, 1]  # l=6
    ]
    
    # Block-level Strassen coefficients
    # X_k blocks: (i_b, j_b) -> coeff
    X_blocks = [
        {(0,0): 1, (1,1): 1}, # k_b=0
        {(1,0): 1, (1,1): 1}, # k_b=1
        {(0,0): 1},           # k_b=2
        {(1,1): 1},           # k_b=3
        {(0,0): 1, (0,1): 1}, # k_b=4
        {(1,0): 1, (0,0): -1},# k_b=5
        {(0,1): 1, (1,1): -1} # k_b=6
    ]
    Y_blocks = [
        {(0,0): 1, (1,1): 1}, # k_b=0
        {(0,0): 1},           # k_b=1
        {(0,1): 1, (1,1): -1},# k_b=2
        {(1,0): 1, (0,0): -1},# k_b=3
        {(1,1): 1},           # k_b=4
        {(0,0): 1, (0,1): 1}, # k_b=5
        {(1,0): 1, (1,1): 1}  # k_b=6
    ]
    C_blocks = [
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}, # k_b=0
        {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1},# k_b=1
        {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}, # k_b=2
        {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}, # k_b=3
        {(0,0): -1,(0,1): 1, (1,0): 0, (1,1): 0}, # k_b=4
        {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}, # k_b=5
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}  # k_b=6
    ]
    
    for k_b in range(7):
        for k_l in range(7):
            col = k_b * 7 + k_l
            # U
            for i_b in range(2):
                for j_b in range(2):
                    coeff_x = X_blocks[k_b].get((i_b, j_b), 0)
                    if coeff_x != 0:
                        for i_l in range(2):
                            for j_l in range(2):
                                i = 2 * i_b + i_l
                                j = 2 * j_b + j_l
                                U[i * m + j][col] += coeff_x * U_loc[k_l][i_l * 2 + j_l]
            # V
            for j_b in range(2):
                for k_b2 in range(2): # use k_b2 to avoid conflict
                    coeff_y = Y_blocks[k_b].get((j_b, k_b2), 0)
                    if coeff_y != 0:
                        for j_l in range(2):
                            for k_l2 in range(2):
                                j = 2 * j_b + j_l
                                k = 2 * k_b2 + k_l2
                                V[j * p + k][col] += coeff_y * V_loc[k_l][j_l * 2 + k_l2]
            # W
            for k_b2 in range(2):
                for i_b in range(2):
                    coeff_c = C_blocks[k_b].get((k_b2, i_b), 0)
                    if coeff_c != 0:
                        for k_l2 in range(2):
                            for i_l in range(2):
                                k = 2 * k_b2 + k_l2
                                i = 2 * i_b + i_l
                                W[k * n + i][col] += coeff_c * W_loc[k_l][k_l2 * 2 + i_l]
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
