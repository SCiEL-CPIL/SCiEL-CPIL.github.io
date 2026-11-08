import time

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s

    n, m, p = 6, 6, 6

    # We'll use 2x2 Strassen blocking: 6 = 2*3
    # 3x3 blocks of 2x2 matrices
    # Each 2x2x2 Strassen uses rank 7
    # Total: 27 block positions * 7 = 189

    # Strassen 2x2x2 rank-1 patterns:
    # C = A @ B where A, B, C are 2x2
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 - M7
    # where:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])

    strassen = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # M1
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # M2
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        # M5
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        # M6
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    block_size = 2
    num_blocks = 3
    total_rank = num_blocks ** 3 * 7  # 27 * 7 = 189

    # Pre-allocate using lists for speed
    # U: shape (n*m, total_rank) = (36, 189)
    # V: shape (m*p, total_rank) = (36, 189)
    # W: shape (p*n, total_rank) = (36, 189)

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                for a_c, b_c, c_c in strassen:
                    for (pi, pj), val in a_c.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c][col] += val
                    for (pi, pj), val in b_c.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col] += val
                    for (pi, pj), val in c_c.items():
                        i = I * block_size + pi
                        k = K * block_size + pj
                        W[k * n + i][col] += val
                    col += 1
                    if time.time() > deadline:
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break
    else:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
