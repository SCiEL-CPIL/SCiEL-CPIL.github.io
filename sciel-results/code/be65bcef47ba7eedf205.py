import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_blocks = p // 2  # 6 blocks of 2 columns each

    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6

    # U coefficients for each Strassen term (A-side)
    U_strassen = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {0: -1, 2: 1},      # M6
        {1: 1, 3: -1},      # M7
    ]

    # V coefficients for each Strassen term (B-side, local indices)
    V_strassen = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1},       # M7
    ]

    # W coefficients for each Strassen term (C-side, local indices)
    W_strassen = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},             # M6
        {0: 1},             # M7
    ]

    # Map local B index to global V row
    def local_b_to_global(local_b, t):
        if local_b == 0:
            return 0 * p + 2 * t
        elif local_b == 1:
            return 0 * p + 2 * t + 1
        elif local_b == 2:
            return 1 * p + 2 * t
        else:
            return 1 * p + 2 * t + 1

    # Map local C index to global W row
    def local_c_to_global(local_c, t):
        if local_c == 0:
            return (2 * t) * n + 0
        elif local_c == 1:
            return (2 * t + 1) * n + 0
        elif local_c == 2:
            return (2 * t) * n + 1
        else:
            return (2 * t + 1) * n + 1

    # Build the rank-42 Strassen decomposition
    rank = num_blocks * 7  # 42

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s

            for a_idx, val in U_strassen[s].items():
                U[a_idx][col] = val

            for local_b, val in V_strassen[s].items():
                global_b = local_b_to_global(local_b, t)
                V[global_b][col] = val

            for local_c, val in W_strassen[s].items():
                global_w = local_c_to_global(local_c, t)
                W[global_w][col] += val

    # Verify the decomposition is exact
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Check: for all a, b, the output C[k,i] = sum_r U[i,r] * V[k,r] * W[k*n+i,r]
    # should equal (A @ B)[i,k]
    # Let's verify with random inputs
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(rank):
        a_lin = U_arr[:, r]  # (n*m,)
        b_lin = V_arr[:, r]  # (m*p,)
        w_lin = W_arr[:, r]  # (p*n,)

        # a_lin maps to A entries: index i*n+j -> A[i,j]
        A_lin = a_lin.reshape(n, m)
        # b_lin maps to B entries: index k*p+l -> B[k,l]
        B_lin = b_lin.reshape(m, p)
        # w_lin maps to W entries: index k*n+i -> W[k*n+i]

        # The scalar product is: (A_lin * A_test).sum() * (B_lin * B_test).sum()
        prod_a = np.sum(A_lin * A_test)
        prod_b = np.sum(B_lin * B_test)

        # Add contribution to C
        for k in range(p):
            for i in range(n):
                C_computed[i, k] += prod_a * prod_b * w_lin[k * n + i]

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to naive rank-48 decomposition
        rank = n * m * p  # 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
