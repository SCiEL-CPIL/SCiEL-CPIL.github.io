import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 140.0  # 10s safety margin

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Naive rank-96 decomposition (baseline)
    naive_rank = n * m * p  # 96
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_naive, "V": V_naive, "W": W_naive}
    best_rank = naive_rank

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(U_arr[i*m+j, :rank], V_arr[j*p+k, :rank] * W_arr[k*n+i, :rank])
        return bool(np.allclose(T, 1.0, atol=1e-8))

    # ---- Build the rank-56+8 = 64 structured decomposition ----
    # 2x2x2 Strassen blocks for i in {0,1}, plus rank-2 (not naive-8) blocks for i=2
    #
    # 1x2x4 block of rank 2:
    #   term a: U[i,j0]=1, V[j0,k0]=1, V[j0,k1]=1, V[j0,k2]=1, V[j0,k3]=1, W[k0,i]=1,W[k1,i]=1,W[k2,i]=1,W[k3,i]=1
    #   term b: U[i,j1]=1, V[j1,k0..k3]=1, W[k0..k3,i]=1
    #   => for (i,j0,k): U=1 * V=1 * W=1 = 1 ; for (i,j1,k): 1*1*1=1 ; for (i,j0) other k: 1*1*1=1
    #   Wait: term a covers (i,j0,k) for all k (V[j0,k]=1 for all k, W[k,i]=1 for all k) -> gives 1 for all k.
    #   term b covers (i,j1,k) for all k -> 1 for all k.
    #   For (i,j0,k): only term a contributes (term b has U[i,j0]=0) -> 1. Good.
    #   For (i,j1,k): only term b -> 1. Good.
    #   So 2 terms suffice for a 1x2x4 block.
    target_rank = 56 + 8  # 4 strassen groups * 7 = 28? No: 2 j_blocks * 2 k_blocks * 2 k_sub = 8 strassen blocks * 7 = 56
                          # 4 naive->rank2 blocks * 2 = 8
    U_new = [[0] * target_rank for _ in range(n * m)]
    V_new = [[0] * target_rank for _ in range(m * p)]
    W_new = [[0] * target_rank for _ in range(p * n)]

    col = 0

    def add_strassen_2x2x2(offset_i, offset_j, offset_k, col_start):
        # 7 Strassen terms for 2x2x2
        col = col_start
        # M1: (A00+A11)(B00+B11) -> C00:+1, C11:+1
        U_new[(offset_i+0)*m+(offset_j+0)][col] += 1
        U_new[(offset_i+1)*m+(offset_j+1)][col] += 1
        V_new[(offset_j+0)*p+(offset_k+0)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+1)][col] += 1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += 1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        # M2: (A10+A11)B00 -> C10:+1, C11:-1
        U_new[(offset_i+1)*m+(offset_j+0)][col] += 1
        U_new[(offset_i+1)*m+(offset_j+1)][col] += 1
        V_new[(offset_j+0)*p+(offset_k+0)][col] += 1
        W_new[(offset_k+0)*n+(offset_i+1)][col] += 1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += -1
        col += 1
        # M3: A00(B10-B11) -> C01:+1, C11:+1
        U_new[(offset_i+0)*m+(offset_j+0)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+0)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+1)][col] += -1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += 1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        # M4: A11(B01-B00) -> C00:+1, C10:+1
        U_new[(offset_i+1)*m+(offset_j+1)][col] += 1
        V_new[(offset_j+0)*p+(offset_k+1)][col] += 1
        V_new[(offset_j+0)*p+(offset_k+0)][col] += -1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += 1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        # M5: (A00+A01)B11 -> C00:-1, C01:+1
        U_new[(offset_i+0)*m+(offset_j+0)][col] += 1
        U_new[(offset_i+0)*m+(offset_j+1)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+1)][col] += 1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += -1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        # M7: (A01-A10)(B10+B11) -> C00:-1, C11:+1
        U_new[(offset_i+0)*m+(offset_j+1)][col] += 1
        U_new[(offset_i+1)*m+(offset_j+0)][col] += -1
        V_new[(offset_j+1)*p+(offset_k+0)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+1)][col] += 1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += -1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        return col

    # 8 Strassen blocks (i in {0,1}, 2 j_blocks, 2 k_blocks, 2 k_sub)
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            for k_sub in [0, 1]:
                offset_i = 0
                offset_j = j_block * 2
                offset_k = k_block * 4 + k_sub * 2
                col = add_strassen_2x2x2(offset_i, offset_j, offset_k, col)

    # 4 rank-2 blocks for i=2 (j_blocks x k_blocks)
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            offset_i = 2
            offset_j = j_block * 2
            offset_k = k_block * 4
            # term a: j = offset_j
            U_new[offset_i*m + offset_j][col] = 1
            for kk in range(4):
                V_new[offset_j*p + (offset_k+kk)][col] = 1
                W_new[(offset_k+kk)*n + offset_i][col] = 1
            col += 1
            # term b: j = offset_j+1
            U_new[offset_i*m + (offset_j+1)][col] = 1
            for kk in range(4):
                V_new[(offset_j+1)*p + (offset_k+kk)][col] = 1
                W_new[(offset_k+kk)*n + offset_i][col] = 1
            col += 1

    if col == target_rank and verify(U_new, V_new, W_new, target_rank):
        best_result = {"U": U_new, "V": V_new, "W": W_new}
        best_rank = target_rank

    if time.time() > DEADLINE - 20:
        return best_result

    # ---- Linear-dependency elimination to reduce rank further ----
    # A column r is redundant if there exist scalars c_s (s != r) such that for every (i,j,k):
    #   U_r[i] V_r[j] W_r[k] = sum_{s != r} c_s U_s[i] V_s[j] W_s[k]
    # Build matrix M: rows = (i,j,k) triples (96), cols = terms (R). M[(i,j,k), r] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    # A column is redundant iff it is in the span of the other columns.
    U_arr = np.array(best_result["U"], dtype=float)
    V_arr = np.array(best_result["V"], dtype=float)
    W_arr = np.array(best_result["W"], dtype=float)
    R = best_rank
    M = np.zeros((n*m*p, R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i*m*p + j*p + k
                for r in range(R):
                    M[row, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]

    def eliminate(M, U, V, W, R):
        """Greedy elimination: repeatedly find a redundant column and drop it."""
        R_cur = R
        kept = list(range(R))
        improved = True
        while improved and R_cur > 1:
            improved = False
            for idx in range(len(kept)):
                r = kept[idx]
                others = [k for k in kept if k != r]
                if len(others) == 0:
                    continue
                sub = M[:, others]
                # Check if M[:, r] is in column space of sub
                # Use rank test: rank([sub | M[:,r]]) == rank(sub)
                try:
                    rank_sub = np.linalg.matrix_rank(sub)
                    combined = np.hstack([sub, M[:, [r]]])
                    rank_comb = np.linalg.matrix_rank(combined)
                except Exception:
                    continue
                if rank_comb == rank_sub:
                    # redundant: drop it
                    kept.remove(r)
                    R_cur -= 1
                    improved = True
                    break
        return kept, R_cur

    kept, R_reduced = eliminate(M, U_arr, V_arr, W_arr, R)
    if R_reduced < R:
        U_red = [[U_arr[i, r] for r in kept] for i in range(n*m)]
        V_red = [[V_arr[i, r] for r in kept] for i in range(m*p)]
        W_red = [[W_arr[i, r] for r in kept] for i in range(p*n)]
        if verify(U_red, V_red, W_red, R_reduced):
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = R_reduced

    if time.time() > DEADLINE - 5:
        return best_result

    # ---- Try to add a new term that lets us drop more (local search) ----
    # Random perturbation: add a random rank-1 term, then re-eliminate.
    rng = np.random.RandomState(42)
    current = best_result
    cur_rank = best_rank
    for trial in range(300):
        if time.time() > DEADLINE - 5:
            break
        U_arr = np.array(current["U"], dtype=float)
        V_arr = np.array(current["V"], dtype=float)
        W_arr = np.array(current["W"], dtype=float)
        R_cur = cur_rank
        # add one random term with small entries
        new_U = U_arr[:, :R_cur].copy()
        new_V = V_arr[:, :R_cur].copy()
        new_W = W_arr[:, :R_cur].copy()
        extra_U = rng.randn(n*m)
        extra_V = rng.randn(m*p)
        extra_W = rng.randn(p*n)
        new_U = np.hstack([new_U, extra_U.reshape(-1,1)])
        new_V = np.hstack([new_V, extra_V.reshape(-1,1)])
        new_W = np.hstack([new_W, extra_W.reshape(-1,1)])
        R_try = R_cur + 1
        M_try = np.zeros((n*m*p, R_try))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i*m*p + j*p + k
                    for r in range(R_try):
                        M_try[row, r] = new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
        kept2, R2 = eliminate(M_try, new_U, new_V, new_W, R_try)
        if R2 < cur_rank:
            U2 = [[new_U[i, r] for r in kept2] for i in range(n*m)]
            V2 = [[new_V[i, r] for r in kept2] for i in range(m*p)]
            W2 = [[new_W[i, r] for r in kept2] for i in range(p*n)]
            if verify(U2, V2, W2, R2):
                current = {"U": U2, "V": V2, "W": W2}
                cur_rank = R2
                best_result = current
                best_rank = R2

    return best_result
