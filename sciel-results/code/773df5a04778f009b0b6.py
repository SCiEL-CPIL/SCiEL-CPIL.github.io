import time

def construct():
    """
    Construct a rank-54 exact tensor decomposition for multiplying a 2x5 matrix
    by a 5x6 matrix.

    Strategy: Partition the 5 columns of A into j-pairs (0,1), (2,3) and j=4,
    and the 6 columns of B into k-pairs (0,1), (2,3), (4,5).
    - For each (j-pair, k-pair) block: use Strassen's 7-term algorithm
      on the 2x2x2 sub-problem (n=2, m=2, p=2). 6 blocks x 7 = 42.
    - For j=4 with each k-pair: use schoolbook (2x1x2). 3 blocks x 4 = 12.
    Total rank = 54.

    Indexing convention:
      U[i*m + j, r]  where i in [0,2), j in [0,5)
      V[j*p + k, r]  where j in [0,5), k in [0,6)
      W[k*n + i, r]  where k in [0,6), i in [0,2)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = 2, 5, 6
    rank = 54

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_rank_step(u_pairs, v_pairs, w_pairs):
        """
        Adds a rank-1 term to the decomposition.
        u_pairs: list of (index_in_U, value)
        v_pairs: list of (index_in_V, value)
        w_pairs: list of (index_in_W, value)
        """
        nonlocal col
        if col >= rank:
            return False
        if time.time() - start_time > deadline:
            return False

        for idx, val in u_pairs:
            U[idx][col] += val
        for idx, val in v_pairs:
            V[idx][col] += val
        for idx, val in w_pairs:
            W[idx][col] += val

        col += 1
        return True

    # Helper to get U index: i * m + j
    def u_idx(i, j):
        return i * m + j

    # Helper to get V index: j * p + k
    def v_idx(j, k):
        return j * p + k

    # Helper to get W index: k * n + i
    def w_idx(k, i):
        return k * n + i

    # Strassen's 7 terms for 2x2x2 matrix multiplication.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # j-pairs for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # k-pairs for all blocks (both Strassen and schoolbook)
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    # --- Part 1: Strassen blocks for j-pairs x k-pairs ---
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            # Local indices for this 2x2x2 block
            # U indices: (i, j) -> i*m + j
            u_a = u_idx(0, j1)  # a = A[0, j1]
            u_b = u_idx(0, j2)  # b = A[0, j2]
            u_c = u_idx(1, j1)  # c = A[1, j1]
            u_d = u_idx(1, j2)  # d = A[1, j2]

            # V indices: (j, k) -> j*p + k
            v_e = v_idx(j1, k1)  # e = B[j1, k1]
            v_f = v_idx(j1, k2)  # f = B[j1, k2]
            v_g = v_idx(j2, k1)  # g = B[j2, k1]
            v_h = v_idx(j2, k2)  # h = B[j2, k2]

            # W indices: (k, i) -> k*n + i
            w_C00 = w_idx(k1, 0)  # C[0, k1]
            w_C01 = w_idx(k2, 0)  # C[0, k2]
            w_C10 = w_idx(k1, 1)  # C[1, k1]
            w_C11 = w_idx(k2, 1)  # C[1, k2]

            # P1 = (a+d)(e+h)
            add_rank_step(
                [(u_a, 1), (u_d, 1)],
                [(v_e, 1), (v_h, 1)],
                [(w_C00, 1), (w_C11, 1)]
            )

            # P2 = (c+d)e
            add_rank_step(
                [(u_c, 1), (u_d, 1)],
                [(v_e, 1)],
                [(w_C10, 1), (w_C11, -1)]
            )

            # P3 = a(f-h)
            add_rank_step(
                [(u_a, 1)],
                [(v_f, 1), (v_h, -1)],
                [(w_C01, 1), (w_C11, 1)]
            )

            # P4 = d(g-e)
            add_rank_step(
                [(u_d, 1)],
                [(v_g, 1), (v_e, -1)],
                [(w_C00, 1), (w_C10, 1)]
            )

            # P5 = (a+b)h
            add_rank_step(
                [(u_a, 1), (u_b, 1)],
                [(v_h, 1)],
                [(w_C00, -1), (w_C01, 1)]
            )

            # P6 = (c-a)(e+f)
            add_rank_step(
                [(u_c, 1), (u_a, -1)],
                [(v_e, 1), (v_f, 1)],
                [(w_C11, 1)]
            )

            # P7 = (b-d)(g+h)
            add_rank_step(
                [(u_b, 1), (u_d, -1)],
                [(v_g, 1), (v_h, 1)],
                [(w_C00, 1)]
            )

    # --- Part 2: Schoolbook for j=4 with each k-pair ---
    # For j=4, A has entries A[0,4] and A[1,4]
    # B has entries B[4,k1] and B[4,k2]
    # C gets contributions to C[i, k1] and C[i, k2] for i in {0,1}
    for k1, k2 in k_pairs:
        # i=0, j=4, k=k1
        add_rank_step(
            [(u_idx(0, 4), 1)],
            [(v_idx(4, k1), 1)],
            [(w_idx(k1, 0), 1)]
        )
        # i=0, j=4, k=k2
        add_rank_step(
            [(u_idx(0, 4), 1)],
            [(v_idx(4, k2), 1)],
            [(w_idx(k2, 0), 1)]
        )
        # i=1, j=4, k=k1
        add_rank_step(
            [(u_idx(1, 4), 1)],
            [(v_idx(4, k1), 1)],
            [(w_idx(k1, 1), 1)]
        )
        # i=1, j=4, k=k2
        add_rank_step(
            [(u_idx(1, 4), 1)],
            [(v_idx(4, k2), 1)],
            [(w_idx(k2, 1), 1)]
        )

    # Verify col == rank
    assert col == rank, f"Expected col={rank}, got col={col}"

    return {"U": U, "V": V, "W": W}
