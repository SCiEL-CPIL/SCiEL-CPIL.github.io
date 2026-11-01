"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, leave margin for safety

    n, m, p = DIMENSIONS

    # Rank 42 decomposition using Block-Strassen
    # n=2, m=4, p=6
    # Blocks: A (1x2 blocks of 2x2), B (2x3 blocks of 2x2), C (1x3 blocks of 2x2)
    # Sum over r=0,1 for each c=0,1,2.
    # 6 products of 2x2 matrices.
    # Each 2x2 product uses Strassen (rank 7).
    # Total rank = 6 * 7 = 42.

    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 multiplication
    # u_part: {(local_i, local_j): coeff}
    # v_part: {(local_i, local_j): coeff}
    # w_part: {(local_i, local_j): coeff}
    
    strassen_terms = [
        # 1: (X00 + X11)(Y00 + Y11) -> Z00 += 1, Z11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 2: (X10 + X11)Y00 -> Z10 += 1, Z11 -= 1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # 3: X00(Y10 - Y11) -> Z01 += 1, Z11 += 1
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # 4: X11(Y01 - Y00) -> Z00 += 1, Z10 += 1
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # 5: (X00 + X01)Y11 -> Z00 -= 1, Z01 += 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # 6: (X10 - X00)(Y00 + Y01) -> Z11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 7: (X11 - X01)(Y10 + Y11) -> Z00 += 1, Z11 += 1
        ({(1,1): 1, (0,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]

    col_idx = 0
    for c in range(p // 2):  # 3 column blocks
        for r in range(m // 2):  # 2 row blocks (summation index)
            for u_part, v_part, w_part in strassen_terms:
                # Update U
                for (li, lj), coeff in u_part.items():
                    global_i = li
                    global_j = r * 2 + lj
                    U[global_i * m + global_j][col_idx] += coeff
                
                # Update V
                for (li, lj), coeff in v_part.items():
                    global_j = r * 2 + li
                    global_k = c * 2 + lj
                    V[global_j * p + global_k][col_idx] += coeff
                
                # Update W
                for (li, lj), coeff in w_part.items():
                    global_i = li
                    global_k = c * 2 + lj
                    W[global_k * n + global_i][col_idx] += coeff
                
                col_idx += 1
                
                if time.time() > deadline:
                    # Fallback to trivial if timeout (should not happen)
                    return construct_trivial()

    return {"U": U, "V": V, "W": W}

def construct_trivial():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
