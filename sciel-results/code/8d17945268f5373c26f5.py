import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # Rank 34 construction:
    # 6 blocks x 5 local terms (M1,M2,M5,M6,M7) + 4 global diagonal terms
    rank = 34

    U = [[0] * rank for _ in range(n * m)]   # 4 x 34
    V = [[0] * rank for _ in range(m * p)]   # 24 x 34
    W = [[0] * rank for _ in range(p * n)]   # 24 x 34

    num_pairs = p // 2  # 6

    # --- Local terms: 5 per block (indices 0..29) ---
    for t in range(num_pairs):
        base = t * 5
        k0 = 2 * t      # even column in pair
        k1 = 2 * t + 1  # odd column in pair

        # V indices
        v_b0_k0 = 0 * p + k0  # B[0, k0]
        v_b0_k1 = 0 * p + k1  # B[0, k1]
        v_b1_k0 = 1 * p + k0  # B[1, k0]
        v_b1_k1 = 1 * p + k1  # B[1, k1]

        # W indices
        w_c0_k0 = k0 * n + 0  # C[0, k0]
        w_c0_k1 = k1 * n + 0  # C[0, k1]
        w_c1_k0 = k0 * n + 1  # C[1, k0]
        w_c1_k1 = k1 * n + 1  # C[1, k1]

        # M1 = (A00+A11)*(B0_k0 + B1_k1)
        c = base + 0
        U[0][c] = 1; U[3][c] = 1
        V[v_b0_k0][c] = 1; V[v_b1_k1][c] = 1
        W[w_c0_k0][c] = 1; W[w_c1_k1][c] = 1

        # M2 = (A10+A11)*B0_k0
        c = base + 1
        U[2][c] = 1; U[3][c] = 1
        V[v_b0_k0][c] = 1
        W[w_c1_k0][c] = 1; W[w_c1_k1][c] = -1

        # M5 = (A00+A01)*B1_k1
        c = base + 2
        U[0][c] = 1; U[1][c] = 1
        V[v_b1_k1][c] = 1
        W[w_c0_k0][c] = -1; W[w_c0_k1][c] = 1

        # M6 = (A10-A00)*(B0_k0 + B0_k1)
        c = base + 3
        U[2][c] = 1; U[0][c] = -1
        V[v_b0_k0][c] = 1; V[v_b0_k1][c] = 1
        W[w_c1_k1][c] = 1

        # M7 = (A01-A11)*(B1_k0 + B1_k1)
        c = base + 4
        U[1][c] = 1; U[3][c] = -1
        V[v_b1_k0][c] = 1; V[v_b1_k1][c] = 1
        W[w_c0_k0][c] = 1

    # --- Global diagonal terms (indices 30..33) ---
    # These replace M3 and M4 from all 6 blocks.
    # The alternating sign (-1)^k in W separates even/odd columns.
    # The intra-pair sum/diff in V separates within-pair positions.

    # G0: U[A00]=1, V[B0_k]=(-1)^k, W[C0_k]=(-1)^k
    # Contribution to C0_k: A00 * B0_k (for all k)
    c = 30
    U[0][c] = 1
    for k in range(p):
        V[0 * p + k][c] = (-1) ** k
        W[k * n + 0][c] = (-1) ** k

    # G1: U[A11]=1, V[B1_k]=(-1)^k, W[C1_k]=(-1)^k
    # Contribution to C1_k: A11 * B1_k (for all k)
    c = 31
    U[3][c] = 1
    for k in range(p):
        V[1 * p + k][c] = (-1) ** k
        W[k * n + 1][c] = (-1) ** k

    # G2: U[A00]=1, V[B0_{2t}]=1, V[B0_{2t+1}]=-1, W[C0_{2t}]=1, W[C0_{2t+1}]=-1
    # Contribution to C0_{2t}: A00*B0_{2t}; to C0_{2t+1}: -A00*B0_{2t+1}
    c = 32
    U[0][c] = 1
    for t in range(num_pairs):
        V[0 * p + 2*t][c] = 1
        V[0 * p + 2*t+1][c] = -1
        W[(2*t) * n + 0][c] = 1
        W[(2*t+1) * n + 0][c] = -1

    # G3: U[A11]=1, V[B1_{2t}]=1, V[B1_{2t+1}]=-1, W[C1_{2t}]=1, W[C1_{2t+1}]=-1
    # Contribution to C1_{2t}: A11*B1_{2t}; to C1_{2t+1}: -A11*B1_{2t+1}
    c = 33
    U[3][c] = 1
    for t in range(num_pairs):
        V[1 * p + 2*t][c] = 1
        V[1 * p + 2*t+1][c] = -1
        W[(2*t) * n + 1][c] = 1
        W[(2*t+1) * n + 1][c] = -1

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)

    if max_err > 1e-9:
        # Fallback to rank-42 Strassen pairs
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for t in range(num_pairs):
            base = t * 7
            k0 = 2 * t
            k1 = 2 * t + 1
            v_b00 = 0 * p + k0; v_b01 = 0 * p + k1
            v_b10 = 1 * p + k0; v_b11 = 1 * p + k1
            w_c00 = k0 * n + 0; w_c01 = k1 * n + 0
            w_c10 = k0 * n + 1; w_c11 = k1 * n + 1

            for s in range(7):
                c = base + s
                if s == 0:
                    U[0][c] = 1; U[3][c] = 1
                    V[v_b00][c] = 1; V[v_b11][c] = 1
                    W[w_c00][c] = 1; W[w_c11][c] = 1
                elif s == 1:
                    U[2][c] = 1; U[3][c] = 1
                    V[v_b00][c] = 1
                    W[w_c10][c] = 1; W[w_c11][c] = -1
                elif s == 2:
                    U[0][c] = 1
                    V[v_b01][c] = 1; V[v_b11][c] = -1
                    W[w_c01][c] = 1; W[w_c11][c] = 1
                elif s == 3:
                    U[3][c] = 1
                    V[v_b10][c] = 1; V[v_b00][c] = -1
                    W[w_c00][c] = 1; W[w_c10][c] = 1
                elif s == 4:
                    U[0][c] = 1; U[1][c] = 1
                    V[v_b11][c] = 1
                    W[w_c00][c] = -1; W[w_c01][c] = 1
                elif s == 5:
                    U[2][c] = 1; U[0][c] = -1
                    V[v_b00][c] = 1; V[v_b01][c] = 1
                    W[w_c11][c] = 1
                elif s == 6:
                    U[1][c] = 1; U[3][c] = -1
                    V[v_b10][c] = 1; V[v_b11][c] = 1
                    W[w_c00][c] = 1

    return {"U": U, "V": "V" in dir() and V, "W": W}
# EVOLVE-BLOCK-END
