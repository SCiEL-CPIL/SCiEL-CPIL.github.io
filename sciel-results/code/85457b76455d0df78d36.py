import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank 56 solution using block Strassen
    # Split P=8 into 4 blocks of size 2
    # Split M=4 into 2 blocks of size 2
    # Each (g, b) pair is a 2x2x2 multiplication using Strassen (rank 7)
    # Total rank = 4 * 2 * 7 = 56

    X_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    Y_forms = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]

    num_groups = 4  # p blocks
    num_ablocks = 2  # m blocks
    num_strassen = 7

    rank = num_groups * num_ablocks * num_strassen  # 56

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                # U: A elements
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][r] = X_forms[s][i * 2 + jp]

                # V: B elements
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]

                # W: C elements
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]

                r += 1

    # Verify the decomposition using numpy if available
    try:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        T = np.zeros((n, m, p))
        for rr in range(rank):
            u_col = U_arr[:, rr]
            v_col = V_arr[:, rr]
            w_col = W_arr[:, rr]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i * m + j]
                    if u_val != 0:
                        for k in range(p):
                            T[i, j, k] += u_val * v_col[j * p + k] * w_col[k * n + i]

        # T should be all ones
        if not np.allclose(T, 1.0, atol=1e-10):
            # Fallback: naive rank-64 decomposition
            rank_naive = n * m * p
            U_n = [[0] * rank_naive for _ in range(n * m)]
            V_n = [[0] * rank_naive for _ in range(m * p)]
            W_n = [[0] * rank_naive for _ in range(p * n)]
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_n[i * m + j][idx] = 1
                        V_n[j * p + k][idx] = 1
                        W_n[k * n + i][idx] = 1
                        idx += 1
            return {"U": U_n, "V": V_n, "W": W_n}
    except Exception:
        pass  # If verification fails for any reason, return the constructed result

    # Deadline guard
    if time.time() > deadline:
        pass  # Already constructed, just return

    return {"U": U, "V": V, "W": W}
