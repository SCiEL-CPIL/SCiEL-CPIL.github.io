import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 10.0

    n, m, p = DIMENSIONS

    # Rank 46 decomposition using Strassen for 6 blocks of 2 columns, and standard for the last 1 column
    rank = 6 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0
    
    # Strassen blocks for k = 0, 2, ..., 10
    for t in range(6):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # B indices
        b_e = 0 * p + k0  # B[0, k0]
        b_f = 0 * p + k1  # B[0, k1]
        b_g = 1 * p + k0  # B[1, k0]
        b_h = 1 * p + k1  # B[1, k1]
        
        # C indices
        c_00 = k0 * n + 0
        c_01 = k1 * n + 0
        c_10 = k0 * n + 1
        c_11 = k1 * n + 1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][r] = 1; U[3][r] = 1
        V[b_e][r] = 1; V[b_h][r] = 1
        W[c_00][r] = 1; W[c_11][r] = 1
        r += 1
        
        # M2 = (A10 + A11) * B0k0
        U[2][r] = 1; U[3][r] = 1
        V[b_e][r] = 1
        W[c_10][r] = 1; W[c_11][r] = -1
        r += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][r] = 1
        V[b_f][r] = 1; V[b_h][r] = -1
        W[c_01][r] = 1; W[c_11][r] = 1
        r += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[3][r] = 1
        V[b_g][r] = 1; V[b_e][r] = -1
        W[c_00][r] = 1; W[c_10][r] = 1
        r += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][r] = 1; U[1][r] = 1
        V[b_h][r] = 1
        W[c_00][r] = -1; W[c_01][r] = 1
        r += 1
        
        # M6 = (A10 - A11) * (B0k0 + B0k1)
        U[2][r] = 1; U[3][r] = -1
        V[b_e][r] = 1; V[b_f][r] = 1
        W[c_11][r] = 1
        r += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][r] = 1; U[3][r] = -1
        V[b_g][r] = 1; V[b_h][r] = 1
        W[c_00][r] = 1
        r += 1

        if time.time() - start_time > deadline:
            break

    # Last column k = 12
    k = 12
    b_e = 0 * p + k
    b_g = 1 * p + k
    c_00 = k * n + 0
    c_10 = k * n + 1

    # T1 = A00 * B0k -> C0k
    U[0][r] = 1
    V[b_e][r] = 1
    W[c_00][r] = 1
    r += 1

    # T2 = A01 * B1k -> C0k
    U[1][r] = 1
    V[b_g][r] = 1
    W[c_00][r] = 1
    r += 1

    # T3 = A10 * B0k -> C1k
    U[2][r] = 1
    V[b_e][r] = 1
    W[c_10][r] = 1
    r += 1

    # T4 = A11 * B1k -> C1k
    U[3][r] = 1
    V[b_g][r] = 1
    W[c_10][r] = 1
    r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
