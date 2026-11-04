import numpy as np
import time

def construct():
    """
    Rank-92 exact tensor decomposition for 4x5x5 matrix multiplication.
    Uses Strassen (rank 7) for 2x2x2 sub-blocks and schoolbook for strips.
    Partition: i:2+2, j:2+2+1, k:2+2+1
    """
    n, m, p = 4, 5, 5
    n_m = n * m
    m_p = m * p
    p_n = p * n

    start_time = time.time()
    deadline = start_time + 110

    U_cols = []
    V_cols = []
    W_cols = []

    def check_deadline():
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")

    def add_term(u_entries, v_entries, w_entries):
        check_deadline()
        u_col = [0.0] * n_m
        v_col = [0.0] * m_p
        w_col = [0.0] * p_n
        for idx, val in u_entries:
            u_col[idx] = val
        for idx, val in v_entries:
            v_col[idx] = val
        for idx, val in w_entries:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def schoolbook_block(i0, j0, k0, si, sj, sk):
        for ii in range(si):
            for jj in range(sj):
                for kk in range(sk):
                    check_deadline()
                    ig = i0 + ii
                    jg = j0 + jj
                    kg = k0 + kk
                    I = ig * m + jg
                    J = jg * p + kg
                    K = kg * n + ig
                    add_term([(I, 1.0)], [(J, 1.0)], [(K, 1.0)])

    def strassen_2x2(i0, j0, k0):
        def gI(i, j): return (i0 + i) * m + (j0 + j)
        def gJ(j, k): return (j0 + j) * p + (k0 + k)
        def gK(k, i): return (k0 + k) * n + (i0 + i)

        add_term([(gI(0,0), 1.0), (gI(1,1), 1.0)],
                 [(gJ(0,0), 1.0), (gJ(1,1), 1.0)],
                 [(gK(0,0), 1.0), (gK(1,1), 1.0)])
        add_term([(gI(1,0), 1.0), (gI(1,1), 1.0)],
                 [(gJ(0,0), 1.0)],
                 [(gK(0,1), 1.0), (gK(1,1), -1.0)])
        add_term([(gI(0,0), 1.0)],
                 [(gJ(0,1), 1.0), (gJ(1,1), -1.0)],
                 [(gK(1,0), 1.0), (gK(1,1), 1.0)])
        add_term([(gI(1,1), 1.0)],
                 [(gJ(1,0), 1.0), (gJ(0,0), -1.0)],
                 [(gK(0,0), 1.0), (gK(0,1), 1.0)])
        add_term([(gI(0,0), 1.0), (gI(0,1), 1.0)],
                 [(gJ(1,1), 1.0)],
                 [(gK(0,0), -1.0), (gK(1,0), 1.0)])
        add_term([(gI(1,0), 1.0), (gI(0,0), -1.0)],
                 [(gJ(0,0), 1.0), (gJ(0,1), 1.0)],
                 [(gK(1,1), 1.0)])
        add_term([(gI(0,1), 1.0), (gI(1,1), -1.0)],
                 [(gJ(1,0), 1.0), (gJ(1,1), 1.0)],
                 [(gK(0,0), 1.0)])

    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 1)]

    for i0, si in i_blocks:
        for j0, sj in j_blocks:
            for k0, sk in k_blocks:
                if si == 2 and sj == 2 and sk == 2:
                    strassen_2x2(i0, j0, k0)
                else:
                    schoolbook_block(i0, j0, k0, si, sj, sk)

    U = np.array(U_cols, dtype=float).T
    V = np.array(V_cols, dtype=float).T
    W = np.array(W_cols, dtype=float).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    construct()
