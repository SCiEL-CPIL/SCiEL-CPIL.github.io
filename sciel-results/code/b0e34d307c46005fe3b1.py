"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 4

    # Strassen-based decomposition: rank 28
    # Partition j (intermediate dim, size 4) into 2 blocks of 2
    # Partition k (output col dim, size 4) into 2 blocks of 2
    # This gives 4 subproblems of 2x2 * 2x2, each with Strassen rank 7
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)

    # Strassen coefficients for 2x2 * 2x2
    # M_f = P_f * Q_f, C = sum_f W_f * M_f
    # u_coeffs[f][pi][pj] = coefficient of A[pi,pj] in P_f
    # v_coeffs[f][pj][qk] = coefficient of B[pj,qk] in Q_f
    # w_coeffs[f][qk][pi] = coefficient of M_f in C[pi,qk]
    u_coeffs = [
        [[1, 0], [0, 1]],   # M1: P = A[0,0]+A[1,1]
        [[0, 0], [1, 1]],   # M2: P = A[1,0]+A[1,1]
        [[1, 0], [0, 0]],   # M3: P = A[0,0]
        [[0, 0], [0, 1]],   # M4: P = A[1,1]
        [[1, 1], [0, 0]],   # M5: P = A[0,0]+A[0,1]
        [[-1, 0], [1, 0]],  # M6: P = A[1,0]-A[0,0]
        [[0, 1], [0, -1]],  # M7: P = A[0,1]-A[1,1]
    ]
    v_coeffs = [
        [[1, 0], [0, 1]],   # M1: Q = B[0,0]+B[1,1]
        [[1, 0], [0, 0]],   # M2: Q = B[0,0]
        [[0, 1], [0, -1]],  # M3: Q = B[0,1]-B[1,1]
        [[-1, 0], [1, 0]],  # M4: Q = B[1,0]-B[0,0]
        [[0, 0], [0, 1]],   # M5: Q = B[1,1]
        [[1, 1], [0, 0]],   # M6: Q = B[0,0]+B[0,1]
        [[0, 0], [1, 1]],   # M7: Q = B[1,0]+B[1,1]
    ]
    w_coeffs = [
        [[1, 0], [0, 1]],   # M1: C[0,0]+C[1,1]
        [[0, 1], [0, -1]],  # M2: C[1,0]-C[1,1]
        [[0, 0], [1, 1]],   # M3: C[0,1]+C[1,1]
        [[1, 1], [0, 0]],   # M4: C[0,0]+C[1,0]
        [[-1, 1], [0, 0]],  # M5: -C[0,0]+C[0,1]
        [[0, 0], [0, 1]],   # M6: C[1,1]
        [[1, 0], [0, 0]],   # M7: C[0,0]
    ]

    # 4 subproblems: (j_block, k_block)
    # s=0: j_block=0, k_block=0
    # s=1: j_block=0, k_block=1
    # s=2: j_block=1, k_block=0
    # s=3: j_block=1, k_block=1
    for s in range(4):
        j_block = s // 2
        k_block = s % 2
        for f in range(7):
            col = s * 7 + f
            # U[i*m + j, col] where i=pi (row of A, 0 or 1), j = 2*j_block + pj
            for pi in range(2):
                for pj in range(2):
                    val = u_coeffs[f][pi][pj]
                    if val != 0:
                        j = 2 * j_block + pj
                        U[pi * m + j][col] += val
            # V[j*p + k, col] where j = 2*j_block + pj, k = 2*k_block + qk
            for pj in range(2):
                for qk in range(2):
                    val = v_coeffs[f][pj][qk]
                    if val != 0:
                        j = 2 * j_block + pj
                        k = 2 * k_block + qk
                        V[j * p + k][col] += val
            # W[k*n + i, col] where k = 2*k_block + qk, i = pi
            for qk in range(2):
                for pi in range(2):
                    val = w_coeffs[f][qk][pi]
                    if val != 0:
                        k = 2 * k_block + qk
                        W[k * n + pi][col] += val

    # Verify the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook
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
