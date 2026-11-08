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

    # Deadline guard
    if time.time() > deadline:
        pass  # Already constructed, just return

    return {"U": U, "V": V, "W": W}
