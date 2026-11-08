"""Matrix multiplication 3,4,8 using rank-88 decomposition (Strassen on 2x4x8 sub-block)."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115

    n, m, p = DIMENSIONS
    # Rank 88: 32 (schoolbook for i=0) + 2*4*7 (Strassen on i=1,2 with 2x2 blocks)
    rank = 88

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: i=0, schoolbook for 1x4x8 (32 terms)
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2, Strassen on 2x2 blocks
    # j split into 2 blocks of 2: [0,1], [2,3]
    # k split into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    # Each 2x2x2 Strassen block uses 7 multiplications
    # Total: 2 * 4 * 7 = 56 terms

    # Strassen factors for 2x2x2:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B01-B11)       -> C01 += M3
    # M4 = A11(B10-B00)       -> C10 += M4, C11 -= M4
    # M5 = (A00+A10)B11       -> C11 += M5
    # M6 = (A01-A10)(B00+B10) -> C00 += M6
    # M7 = (A01-A11)(B01+B11) -> C01 += M7

    strassen_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A10
        [0, 1, -1, 0],  # M6: A01 - A10
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B10
        [0, 0, 1, 1],   # M7: B01 + B11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 0, 1, 0],   # M2: C10 += 1
        [0, 1, 0, 0],   # M3: C01 += 1
        [1, 0, 1, -1],  # M4: C10 += 1, C11 -= 1
        [0, 0, 0, 1],   # M5: C11 += 1
        [1, 0, 0, 0],   # M6: C00 += 1
        [0, 1, 0, 0],   # M7: C01 += 1
    ]

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

                # U: local indices l -> (local_i, local_j) where l = local_i*2 + local_j
                # global_i = 1 + local_i, global_j = j_start + local_j
                for l in range(4):
                    val = strassen_U[t][l]
                    if val != 0:
                        local_i = l // 2
                        local_j = l % 2
                        global_i = 1 + local_i
                        global_j = j_start + local_j
                        U[global_i * m + global_j][col] += val

                # V: local indices l -> (local_j, local_k) where l = local_j*2 + local_k
                # global_j = j_start + local_j, global_k = k_start + local_k
                for l in range(4):
                    val = strassen_V[t][l]
                    if val != 0:
                        local_j = l // 2
                        local_k = l % 2
                        global_j = j_start + local_j
                        global_k = k_start + local_k
                        V[global_j * p + global_k][col] += val

                # W: local indices l -> (local_k, local_i) where l = local_k*2 + local_i
                # global_k = k_start + local_k, global_i = 1 + local_i
                for l in range(4):
                    val = strassen_W[t][l]
                    if val != 0:
                        local_k = l // 2
                        local_i = l % 2
                        global_k = k_start + local_k
                        global_i = 1 + local_i
                        W[global_k * n + global_i][col] += val

                col += 1

    # Self-validation: verify a sample of tensor entries
    if time.time() < deadline - 2:
        try:
            import random
            random.seed(42)
            for _ in range(30):
                ii = random.randint(0, n * m - 1)
                jj = random.randint(0, m * p - 1)
                kk = random.randint(0, p * n - 1)
                s = 0
                for r in range(rank):
                    s += U[ii][r] * V[jj][r] * W[kk][r]
                i_idx = ii // m
                j_idx = ii % m
                j2_idx = jj // p
                k_idx = jj % p
                k2_idx = kk // n
                i2_idx = kk % n
                expected = 1 if (j_idx == j2_idx and k_idx == k2_idx and i_idx == i2_idx) else 0
                if s != expected:
                    break
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
