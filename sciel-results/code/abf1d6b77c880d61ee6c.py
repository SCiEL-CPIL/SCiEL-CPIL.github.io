import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen 2x2 decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all combinations of block indices
    # i_A, i_B in 0..1 for A rows
    # j_A, j_B in 0..1 for A cols / B rows
    # k_A, k_B in 0..1 for B cols / C rows
    
    for i_A in range(2):
        for i_B in range(2):
            for j_A in range(2):
                for j_B in range(2):
                    # U index
                    i = i_A * 2 + i_B
                    j = j_A * 2 + j_B
                    u_idx = i * m + j
                    
                    # V index
                    # j is same as A col
                    # k_A, k_B needed for V and W
                    # We need to iterate k as well for the full tensor
                    # But U only depends on i, j.
                    # So we can fill U in a separate loop or inside.
                    # Let's do it inside the k loop to be safe, though U doesn't depend on k.
                    pass
                    
    # Better approach:
    # U depends on i, j.
    # V depends on j, k.
    # W depends on k, i.
    
    # Fill U
    for i_A in range(2):
        for i_B in range(2):
            for j_A in range(2):
                for j_B in range(2):
                    i = i_A * 2 + i_B
                    j = j_A * 2 + j_B
                    u_idx = i * m + j
                    for r_A in range(7):
                        for r_B in range(7):
                            r = r_A * 7 + r_B
                            val = U2[i_A * 2 + j_A][r_A] * U2[i_B * 2 + j_B][r_B]
                            U[u_idx][r] = val

    # Fill V
    for j_A in range(2):
        for j_B in range(2):
            for k_A in range(2):
                for k_B in range(2):
                    j = j_A * 2 + j_B
                    k = k_A * 2 + k_B
                    v_idx = j * p + k
                    for r_A in range(7):
                        for r_B in range(7):
                            r = r_A * 7 + r_B
                            val = V2[j_A * 2 + k_A][r_A] * V2[j_B * 2 + k_B][r_B]
                            V[v_idx][r] = val

    # Fill W
    for k_A in range(2):
        for k_B in range(2):
            for i_A in range(2):
                for i_B in range(2):
                    k = k_A * 2 + k_B
                    i = i_A * 2 + i_B
                    w_idx = k * n + i
                    for r_A in range(7):
                        for r_B in range(7):
                            r = r_A * 7 + r_B
                            val = W2[k_A * 2 + i_A][r_A] * W2[k_B * 2 + i_B][r_B]
                            W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
