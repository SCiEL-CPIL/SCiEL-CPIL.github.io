import random
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 110.0

    n, m, p = 2, 4, 8

    # Rank-56 Strassen decomposition:
    # 2 (A rows) x 4 (shared) x 8 (B cols)
    # Split into 4 column groups of size 2, 2 row-blocks of size 2, 7 Strassen terms
    # Total rank = 4 * 2 * 7 = 56

    # Standard Strassen for 2x2 * 2x2:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B10-B11)
    # M3 = A11*(B01-B10)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B10)
    # M6 = (A01-A11)*(B01+B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M1 + M4
    # C10 = M2 + M5
    # C11 = M0 - M1 + M2 + M5

    # X forms: coefficients for A block (2x2 = [i0,j0; i1,j1]) indexed by [i*2+jp]
    X_forms = [
        [1, 0, 0, 1],   # M0: A00+A11
        [0, 0, 1, 1],   # M1: A10+A11
        [1, 0, 0, 0],   # M2: A00
        [0, 0, 0, 1],   # M3: A11
        [1, 1, 0, 0],   # M4: A00+A01
        [-1, 0, 1, 0],  # M5: A10-A00
        [0, 1, 0, -1],  # M6: A01-A11
    ]

    # Y forms: coefficients for B block (2x2 = [j0,k0; j1,k1]) indexed by [jp*2+kp]
    Y_forms = [
        [1, 0, 0, 1],   # M0: B00+B11
        [1, 0, 0, 0],   # M1: B00
        [0, 1, 0, -1],  # M2: B10-B11
        [-1, 0, 1, 0],  # M3: B01-B10
        [0, 0, 0, 1],   # M4: B11
        [1, 1, 0, 0],   # M5: B00+B10
        [0, 0, 1, 1],   # M6: B01+B11
    ]

    # Z combos: coefficients for C output (2x2 = [k0,i0; k1,i1]) indexed by [i*2+kp]
    # C00 = M0 + M3 - M4 + M6
    # C01 = M1 + M4
    # C10 = M2 + M5
    # C11 = M0 - M1 + M2 + M5
    #
    # Index [i*2+kp]: i=0,kp=0 -> C[0,0]; i=0,kp=1 -> C[0,1]; i=1,kp=0 -> C[1,0]; i=1,kp=1 -> C[1,1]
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0] = M0 + M3 - M4 + M6
        [0, 1, 0, 0, 1, 0, 0],    # C[0,1] = M1 + M4
        [0, 0, 1, 0, 0, 1, 0],    # C[1,0] = M2 + M5
        [1, -1, 1, 0, 0, 1, 0],   # C[1,1] = M0 - M1 + M2 + M5
    ]

    num_groups = 4      # groups of 2 columns in C (8/2)
    num_ablocks = 2     # blocks of 2 columns in A (4/2)
    num_strassen = 7
    rank = num_groups * num_ablocks * num_strassen  # 56

    U = [[0] * rank for _ in range(n * m)]   # (8, 56)
    V = [[0] * rank for _ in range(m * p)]   # (32, 56)
    W = [[0] * rank for _ in range(p * n)]   # (16, 56)

    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                # U: A[i, j] where i in [0,2), j = b*2 + jp
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][r] = X_forms[s][i * 2 + jp]

                # V: B[j, k] where j = b*2 + jp, k = 2*g + kp
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]

                # W: C[k, i] where k = 2*g + kp, i in [0,2)
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]

                r += 1

    if time.time() > DEADLINE:
        # Should not happen for this fixed computation
        pass

    return {"U": U, "V": V, "W": W}

if __name__ == '__main__':
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']

    n, m, p = 2, 4, 8
    assert len(U) == n*m, f"U shape error: {len(U)} != {n*m}"
    assert len(U[0]) == len(V[0]), "Rank mismatch U V"
    assert len(V) == m*p, f"V shape error: {len(V)} != {m*p}"
    assert len(W) == p*n, f"W shape error: {len(W)} != {p*n}"
    assert len(W[0]) == len(U[0]), "Rank mismatch W U"

    # Verify correctness with random test
    import random
    random.seed(42)
    A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]

    # Compute C = A * B
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]

    # Compute C' using the decomposition
    C_prime = [[0] * p for _ in range(n)]
    for r in range(len(U[0])):
        u_val = sum(U[i*m + j][r] * A[i][j] for i in range(n) for j in range(m))
        v_val = sum(V[j*p + k][r] * B[j][k] for j in range(m) for k in range(p))
        prod = u_val * v_val
        for k in range(p):
            for i in range(n):
                C_prime[i][k] += W[k*n + i][r] * prod

    for i in range(n):
        for k in range(p):
            if C[i][k] != C_prime[i][k]:
                print(f"Mismatch at C[{i}][{k}]: {C[i][k]} vs {C_prime[i][k]}")
                sys.exit(1)

    print(f"Verified correct. Rank = {len(U[0])}")
