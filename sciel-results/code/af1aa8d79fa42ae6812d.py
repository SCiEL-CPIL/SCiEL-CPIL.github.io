"""Tensor decomposition for 2x3x8 matrix multiplication using a rank-44 construction."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition vectors
    # u: (i, j) in 0..1, 0..1 -> flattened 0..3
    # v: (j, k) in 0..1, 0..1 -> flattened 0..3
    # w: (k, i) in 0..1, 0..1 -> flattened 0..3
    # Note: w index in global W is k*n + i. Here n=2.
    # Local w index k'*2 + i'.
    
    strassen_u = [
        [1, 0, 1, 0], # m1: a+d
        [0, 0, 1, 1], # m2: c+d
        [1, 0, 0, 0], # m3: a
        [0, 0, 0, 1], # m4: d
        [1, 1, 0, 0], # m5: a+b
        [-1, 0, 1, 0],# m6: c-a
        [0, 1, 0, -1] # m7: b-d
    ]
    strassen_v = [
        [1, 1, 0, 0], # m1: e+h
        [1, 0, 0, 0], # m2: e
        [0, 1, 0, -1],# m3: f-h
        [0, 0, 1, -1],# m4: g-e
        [0, 0, 0, 1], # m5: h
        [1, 1, 0, 0], # m6: e+f
        [0, 0, 1, 1]  # m7: g+h
    ]
    strassen_w = [
        [1, 0, 0, 1], # m1: C11+=1, C22+=1
        [0, 0, 1, -1],# m2: C21+=1, C22-=1
        [0, 1, 0, 1], # m3: C12+=1, C22+=1
        [1, 0, 1, 0], # m4: C11+=1, C21+=1
        [-1, 1, 0, 0],# m5: C11-=1, C12+=1
        [0, 0, 0, 1], # m6: C22+=1
        [1, 0, 0, 0]  # m7: C11+=1
    ]

    col = 0
    # Process 4 blocks of 2x2x2
    for q in range(4):
        for r in range(7):
            u_vec = strassen_u[r]
            v_vec = strassen_v[r]
            w_vec = strassen_w[r]
            
            # Map u (2x2) to U (2x3)
            # u_vec index: i'*2 + j'
            for i_prime in range(2):
                for j_prime in range(2):
                    val = u_vec[i_prime * 2 + j_prime]
                    if val != 0:
                        i_global = i_prime
                        j_global = j_prime
                        U[i_global * m + j_global][col] = val

            # Map v (2x2) to V (3x8)
            # v_vec index: j'*2 + k'
            for j_prime in range(2):
                for k_prime in range(2):
                    val = v_vec[j_prime * 2 + k_prime]
                    if val != 0:
                        j_global = j_prime
                        k_global = q * 2 + k_prime
                        V[j_global * p + k_global][col] = val

            # Map w (2x2) to W (8x2)
            # w_vec index: k'*2 + i'
            for k_prime in range(2):
                for i_prime in range(2):
                    val = w_vec[k_prime * 2 + i_prime]
                    if val != 0:
                        k_global = q * 2 + k_prime
                        i_global = i_prime
                        W[k_global * n + i_global][col] = val
            
            col += 1

    # Process 2x1x8 part (A col 3, B row 3)
    # A index (i, 2), B index (2, k), W index (k, i)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
