import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_left * B_left (j=0)
    # r=0
    U[0][0] = 1 # i=0, j=0
    for k in range(p):
        V[k][0] = 1 # j=0, k
        W[2*k][0] = 1 # k, i=0
    # r=1
    U[3][1] = 1 # i=1, j=0
    for k in range(p):
        V[k][1] = 1
        W[2*k+1][1] = 1 # k, i=1
        
    # Part 2: A_right * B_right (j=1,2)
    # Strassen coefficients
    # u[i][j], v[j][k], w[k][i] for i,j,k in 0..1
    u_coeffs = [
        [[1, 0], [0, 1]], # s=0
        [[0, 0], [1, 1]], # s=1
        [[1, 0], [0, 0]], # s=2
        [[0, 0], [0, 1]], # s=3
        [[1, 1], [0, 0]], # s=4
        [[-1, 0], [1, 0]],# s=5
        [[0, 1], [-1, 0]] # s=6
    ]
    v_coeffs = [
        [[1, 0], [0, 1]], # s=0
        [[1, 0], [0, 0]], # s=1
        [[0, 1], [-1, 0]],# s=2
        [[-1, 0], [1, 0]],# s=3
        [[0, 0], [0, 1]], # s=4
        [[1, 1], [0, 0]], # s=5
        [[0, 1], [0, 1]]  # s=6
    ]
    w_coeffs = [
        [[1, 1], [1, 1]], # s=0: w00, w01, w10, w11
        [[0, 0], [1, -1]],# s=1
        [[0, 1], [0, 1]], # s=2
        [[1, 0], [1, 0]], # s=3
        [[-1, 1], [0, 0]],# s=4
        [[0, 0], [0, 1]], # s=5
        [[1, 0], [0, 0]]  # s=6
    ]
    
    for b in range(3):
        r_base = 2 + 7 * b
        for s in range(7):
            r = r_base + s
            # Fill U
            for i in range(2):
                for j_prime in range(2):
                    val = u_coeffs[s][i][j_prime]
                    if val != 0:
                        j_global = j_prime + 1
                        u_idx = i * m + j_global
                        U[u_idx][r] += val
            # Fill V
            for j_prime in range(2):
                for k_prime in range(2):
                    val = v_coeffs[s][j_prime][k_prime]
                    if val != 0:
                        j_global = j_prime + 1
                        k_global = 2 * b + k_prime
                        v_idx = j_global * p + k_global
                        V[v_idx][r] += val
            # Fill W
            for k_prime in range(2):
                for i in range(2):
                    val = w_coeffs[s][k_prime][i]
                    if val != 0:
                        k_global = 2 * b + k_prime
                        w_idx = k_global * n + i
                        W[w_idx][r] += val
                        
    return {"U": U, "V": V, "W": W}
