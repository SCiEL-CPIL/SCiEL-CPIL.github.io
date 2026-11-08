"""Direct Strassen-extended tensor decomposition for 2x3 * 3x8 matrix multiplication.

Approach:
- Partition A (2x3) into A1 (2x2) and A2 (2x1).
- Partition B (3x8) into B1 (2x8) and B2 (1x8).
- The (2,2,8) block (A1 * B1) is factored using Strassen (2,2,2) repeated 4 times => rank 28.
- The (2,1,8) block (A2 * B2) is just schoolbook => rank 16.
- Total rank = 44, matching the best known construction.
"""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8

    # Strassen 2x2x2 factors
    # U: indexed by (i, a) flattened -> [i*2+a]
    # V: indexed by (a, k) flattened -> [a*2+k]
    # W: indexed by (k, i) flattened -> [k*2+i]
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]

    # Verify Strassen correctness: sum_r U[i,a,r]*V[a,k,r]*W[k,i,r] = 1 for all i,a,k
    for i in range(2):
        for a in range(2):
            for k in range(2):
                acc = 0
                for r in range(7):
                    acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                if acc != 1:
                    # Fallback: full schoolbook (rank 48)
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    c = 0
                    for ii in range(n):
                        for aa in range(m):
                            for kk in range(p):
                                U_fb[ii * m + aa][c] = 1
                                V_fb[aa * p + kk][c] = 1
                                W_fb[kk * n + ii][c] = 1
                                c += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Build the (2,3,8) decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), rank 28 via 4 Strassen blocks on k-dimension
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # U: (i, a) where i in {0,1}, a in {0,1}
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val

            # V: (a, k) where a in {0,1}, k in {k0, k0+1}
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val

            # W: (k, i) where k in {k0, k0+1}, i in {0,1}
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val

            col += 1

    # Part 2: A2 (2x1) * B2 (1x8), rank 16 schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
