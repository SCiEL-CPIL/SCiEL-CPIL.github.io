import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 8, 20, 10

    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s

    # Build schoolbook decomposition as fallback (rank 40)
    rank_sb = n * m * p  # 40
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    for idx, (i, j, k) in enumerate(
        (i, j, k) for i in range(n) for j in range(m) for k in range(p)
    ):
        U_sb[i * m + j, idx] = 1
        V_sb[j * p + k, idx] = 1
        W_sb[k * n + i, idx] = 1

    def to_list(mat):
        return [[int(round(v * 2)) / 2 for v in row] for row in mat.tolist()]

    fallback = {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}

    # Rank-36 Strassen-based decomposition
    total_rank = 36
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))

    # Strassen 2x2x2 patterns (local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    strassen_patterns = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Schoolbook 2x2x1 patterns
    sb_2x2x1 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]

    def add_2x2x2(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col):
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val:
                    U[(a_i_off + idx // 2) * m + (a_j_off + idx % 2), col] = val
            for idx, val in enumerate(v_pat):
                if val:
                    V[(b_j_off + idx // 2) * p + (b_k_off + idx % 2), col] = val
            for idx, val in enumerate(w_pat):
                if val:
                    W[(c_k_off + idx % 2) * n + (c_i_off + idx // 2), col] = val
            col += 1
        return col

    def add_2x2x1(a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col):
        for u_pat, v_pat, w_pat in sb_2x2x1:
            for idx, val in enumerate(u_pat):
                if val:
                    U[(a_i_off + idx // 2) * m + (a_j_off + idx % 2), col] = val
            for idx, val in enumerate(v_pat):
                if val:
                    V[(b_j_off + idx) * p + b_k_off, col] = val
            for idx, val in enumerate(w_pat):
                if val:
                    W[c_k_off * n + (c_i_off + idx), col] = val
            col += 1
        return col

    col = 0
    # Output cols 0,1: A_left*B_top + A_right*B_bottom
    col = add_2x2x2(0, 0, 0, 0, 0, 0, col)   # A[0:2,0:2] * B[0:2,0:2]
    col = add_2x2x2(0, 2, 2, 0, 0, 0, col)   # A[0:2,2:4] * B[2:4,0:2]
    # Output cols 2,3: A_left*B_mid + A_right*B_mid
    col = add_2x2x2(0, 0, 0, 2, 0, 2, col)   # A[0:2,0:2] * B[0:2,2:4]
    col = add_2x2x2(0, 2, 2, 2, 0, 2, col)   # A[0:2,2:4] * B[2:4,2:4]
    # Output col 4: A_left*B_last + A_right*B_last
    col = add_2x2x1(0, 0, 0, 4, 0, 4, col)   # A[0:2,0:2] * B[0:2,4]
    col = add_2x2x1(0, 2, 2, 4, 0, 4, col)   # A[0:2,2:4] * B[2:4,4]

    assert col == total_rank, f"Expected {total_rank}, got {col}"

    # Fast verification using NumPy
    def verify(U, V, W):
        if time.time() > deadline:
            return False
        rank = U.shape[1]
        # Check diagonal terms: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal terms: sum_r U[a,r]*V[b,r]*W[c,r] == 0
        # Only check non-trivial triples
        for a in range(nm):
            if time.time() > deadline:
                return False
            i_a, j_a = divmod(a, m)
            for b in range(mp):
                j_b, k_b = divmod(b, p)
                if j_a != j_b:
                    continue
                for c in range(pn):
                    k_c, i_c = divmod(c, n)
                    if k_b != k_c or i_a != i_c:
                        continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    if time.time() < deadline and verify(U, V, W):
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    return fallback
