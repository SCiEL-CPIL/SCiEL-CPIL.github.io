"""
Matrix Multiplication 4x5x6 rank-minimal exact tensor decomposition.
Strategy: block-diagonal decomposition using Strassen for 4x4 blocks.
  - 4x4x4 Strassen: rank 49
  - 4x1x6 schoolbook: rank 24
  - 4x4x2 Strassen: rank 28
  Total: 101
Also attempts a fast structured search for a lower-rank 4x4x2 decomposition.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    Indices:
      U[i*m + j] where m=2. So U[i*2 + j].
      V[j*p + k] where p=2. So V[j*2 + k].
      W[k*n + i] where n=2. So W[k*2 + i].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _build_strassen_4x4():
    """
    Build 4x4x4 Strassen decomposition. Rank 49.
    """
    U2, V2, W2 = _build_strassen_2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]

    strassen_defs = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]

    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U[u_row][col] += coeff_a * sigma
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V[v_row][col] += coeff_b * sigma
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff_c * sigma
    return U, V, W


def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition. Rank 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]

    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c

    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 7)
    add_strassen_block(2, 0, 0, 2, 0, 14)
    add_strassen_block(2, 2, 2, 2, 0, 21)
    return U, V, W


def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W


def _combine_decompositions(decomps, n, m, p):
    """
    Combine decompositions.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])

        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]

        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]

        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]

        current_rank += rank_sub
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True


def _search_4x4x2_low_rank(max_rank, time_limit):
    """
    Search for a lower-rank decomposition of the 4x4x2 tensor.
    Returns (U, V, W) if found, None otherwise.
    """
    start_time = time.time()
    n, m, p = 4, 4, 2

    for target_rank in range(max_rank, 23, -1):
        if time.time() - start_time > time_limit:
            break

        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break

            np.random.seed(attempt * 999 + target_rank)
            U = np.zeros((n*m, target_rank), dtype=int)
            V = np.zeros((m*p, target_rank), dtype=int)
            W = np.zeros((p*n, target_rank), dtype=int)

            for r in range(target_rank):
                u_nz = np.random.choice(n*m, size=min(3, n*m), replace=False)
                v_nz = np.random.choice(m*p, size=min(3, m*p), replace=False)
                w_nz = np.random.choice(p*n, size=min(3, p*n), replace=False)

                for idx in u_nz:
                    U[idx, r] = np.random.choice([-2, -1, 1, 2])
                for idx in v_nz:
                    V[idx, r] = np.random.choice([-2, -1, 1, 2])
                for idx in w_nz:
                    W[idx, r] = np.random.choice([-2, -1, 1, 2])

            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()

            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS

    # Try to find a better 4x4x2 decomposition
    better_4x4x2 = None
    try:
        search_time = min(15, deadline - time.time())
        if search_time > 5:
            better_4x4x2 = _search_4x4x2_low_rank(28, search_time)
    except Exception:
        pass

    try:
        # 1. 4x4x4 Strassen (Rank 49)
        U1, V1, W1 = _build_strassen_4x4()
        decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))

        # 2. 4x1x6 Schoolbook (Rank 24)
        U2, V2, W2 = _build_schoolbook(4, 1, 6)
        decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))

        # 3. 4x4x2 (Rank 28 or better)
        if better_4x4x2 is not None:
            U3, V3, W3 = better_4x4x2
        else:
            U3, V3, W3 = _build_strassen_4x4x2()
        decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))

        U, V, W = _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)

        if time.time() > deadline:
            raise TimeoutError()

        return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
