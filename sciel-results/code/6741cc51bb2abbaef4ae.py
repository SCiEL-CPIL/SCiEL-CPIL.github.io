import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 108.0

    n, m, p = 6, 6, 6
    nm, mp, pn = 36, 36, 36

    # Schoolbook rank-216 baseline
    rank_school = 216
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_s[i * m + j, col] = 1.0
                V_s[j * p + k, col] = 1.0
                W_s[k * n + i, col] = 1.0

    def check(U, V, W):
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        Ug = U.reshape(6, 6, -1)
        Vg = V.reshape(6, 6, -1)
        Wg = W.reshape(6, 6, -1)
        rec = np.einsum('ijr,jkr,kir->ijk', Ug, Vg, Wg, optimize=True)
        return np.allclose(rec, np.ones((6, 6, 6)), atol=1e-9)

    # --- Strategy: 6x6 = 2x2 blocks of 3x3, Strassen outer (rank 7) x schoolbook inner (rank 27) = 189 ---
    n3 = 3
    r3_school = 27
    U3 = np.zeros((9, r3_school))
    V3 = np.zeros((9, r3_school))
    W3 = np.zeros((9, r3_school))
    for i3 in range(n3):
        for j3 in range(n3):
            for k3 in range(n3):
                col3 = i3 * 9 + j3 * 3 + k3
                U3[i3 * 3 + j3, col3] = 1.0
                V3[j3 * 3 + k3, col3] = 1.0
                W3[k3 * 3 + i3, col3] = 1.0

    total_rank = 7 * r3_school  # 189
    U6 = np.zeros((nm, total_rank))
    V6 = np.zeros((mp, total_rank))
    W6 = np.zeros((pn, total_rank))

    def add_term(start, a_blks, b_blks, c_blks):
        for r in range(r3_school):
            um = U3[:, r].reshape(3, 3)
            vm = V3[:, r].reshape(3, 3)
            wm = W3[:, r].reshape(3, 3)
            for bi, cf in a_blks:
                br, bc = bi // 2, bi % 2
                for ii in range(3):
                    for jj in range(3):
                        v = cf * um[ii, jj]
                        if v != 0:
                            U6[(br * 3 + ii) * 6 + (bc * 3 + jj), start + r] += v
            for bi, cf in b_blks:
                br, bc = bi // 2, bi % 2
                for jj in range(3):
                    for kk in range(3):
                        v = cf * vm[jj, kk]
                        if v != 0:
                            V6[(br * 3 + jj) * 6 + (bc * 3 + kk), start + r] += v
            for bi, cf in c_blks:
                br, bc = bi // 2, bi % 2
                for kk in range(3):
                    for ii in range(3):
                        v = cf * wm[kk, ii]
                        if v != 0:
                            W6[(bc * 3 + kk) * 6 + (br * 3 + ii), start + r] += v

    # Strassen 2x2 block formulas
    add_term(0,
             [(0, 1), (3, 1)],
             [(0, 1), (3, 1)],
             [(0, 1), (3, 1)])
    add_term(r3_school,
             [(2, 1), (3, 1)],
             [(0, 1)],
             [(2, 1), (3, -1)])
    add_term(2 * r3_school,
             [(0, 1)],
             [(1, 1), (3, -1)],
             [(1, 1), (3, 1)])
    add_term(3 * r3_school,
             [(3, 1)],
             [(2, 1), (0, -1)],
             [(0, 1), (2, 1)])
    add_term(4 * r3_school,
             [(0, 1), (1, 1)],
             [(3, 1)],
             [(1, 1), (0, -1)])
    add_term(5 * r3_school,
             [(2, 1), (0, -1)],
             [(0, 1), (1, 1)],
             [(3, 1)])
    add_term(6 * r3_school,
             [(1, 1), (3, -1)],
             [(2, 1), (3, 1)],
             [(0, 1)])

    best_U, best_V, best_W = U_s, V_s, W_s
    if check(U6, V6, W6):
        best_U, best_V, best_W = U6, V6, W6

    # --- Refinement: greedy column removal to find minimal rank subset ---
    if check(best_U, best_V, best_W) and time.time() < deadline - 15.0:
        try:
            R = best_U.shape[1]
            keep = list(range(R))
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                for r in list(keep):
                    if len(keep) <= 1:
                        break
                    new_keep = [k for k in keep if k != r]
                    U_t = best_U[:, new_keep]
                    V_t = best_V[:, new_keep]
                    W_t = best_W[:, new_keep]
                    if check(U_t, V_t, W_t):
                        keep = new_keep
                        best_U, best_V, best_W = U_t, V_t, W_t
                        improved = True
                        break
        except Exception:
            pass

    # --- Second refinement: try to merge columns via linear combinations (rank reduction) ---
    if check(best_U, best_V, best_W) and time.time() < deadline - 5.0:
        try:
            R = best_U.shape[1]
            if R > 100:
                for i in range(min(R - 1, 50)):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, min(R, i + 50)):
                        new_keep = [k for k in range(R) if k != i and k != j]
                        U_t = best_U[:, new_keep]
                        V_t = best_V[:, new_keep]
                        W_t = best_W[:, new_keep]
                        if check(U_t, V_t, W_t):
                            best_U, best_V, best_W = U_t, V_t, W_t
                            R = len(new_keep)
                            break
        except Exception:
            pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
