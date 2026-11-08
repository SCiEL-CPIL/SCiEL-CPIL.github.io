import time
import numpy as np

DIMENSIONS = [3, 4, 6]


def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    n, m, p = DIMENSIONS

    def verify_decomp(U, V, W, rank):
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def make_builder(rank):
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_iv, v_iv, w_iv):
            if cur[0] >= rank:
                return False
            for idx, val in u_iv:
                U[idx][cur[0]] += val
            for idx, val in v_iv:
                V[idx][cur[0]] += val
            for idx, val in w_iv:
                W[idx][cur[0]] += val
            cur[0] += 1
            return True

        return U, V, W, cur, add_rank1

    def build_rank66():
        """p=4+2 split, n=1+2 per block, Strassen on 2x2x2."""
        rank = 66
        U, V, W, cur, add_rank1 = make_builder(rank)

        def add_sb(i0, j0, k0, nn, mm, pp):
            for ii in range(nn):
                for jj in range(mm):
                    for kk in range(pp):
                        add_rank1([((i0 + ii) * m + (j0 + jj), 1)],
                                  [((j0 + jj) * p + (k0 + kk), 1)],
                                  [((k0 + kk) * n + (i0 + ii), 1)])

        def add_ssr(i0, j0, k0):
            U2 = lambda i, j: (i0 + i) * m + (j0 + j)
            V2 = lambda j, k: (j0 + j) * p + (k0 + k)
            W2 = lambda i, k: (k0 + k) * n + (i0 + i)
            add_rank1([(U2(0,0),1),(U2(1,1),1)], [(V2(0,0),1),(V2(1,1),1)], [(W2(0,0),1),(W2(1,1),1)])
            add_rank1([(U2(1,0),1),(U2(1,1),1)], [(V2(0,0),1)], [(W2(1,0),1),(W2(1,1),-1)])
            add_rank1([(U2(0,0),1)], [(V2(0,1),1),(V2(1,1),-1)], [(W2(0,1),1),(W2(1,1),1)])
            add_rank1([(U2(1,1),1)], [(V2(1,0),1),(V2(0,0),-1)], [(W2(0,0),1),(W2(1,0),1)])
            add_rank1([(U2(0,0),1),(U2(0,1),1)], [(V2(1,1),1)], [(W2(0,0),-1),(W2(0,1),1)])
            add_rank1([(U2(1,0),1),(U2(0,0),-1)], [(V2(0,0),1),(V2(0,1),1)], [(W2(1,1),1)])
            add_rank1([(U2(0,1),1),(U2(1,1),-1)], [(V2(1,0),1),(V2(1,1),1)], [(W2(0,0),1)])

        add_sb(0, 0, 0, 1, 4, 4)
        add_ssr(1, 0, 0)
        add_ssr(1, 2, 0)
        add_ssr(1, 0, 2)
        add_ssr(1, 2, 2)
        add_sb(0, 0, 4, 1, 4, 2)
        add_ssr(1, 0, 4)
        add_ssr(1, 2, 4)

        return {"U": U, "V": V, "W": W}

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def greedy_search(bound):
        """Greedy set-cover over binary rank-1 terms; returns exact decomposition or None."""
        T = n * m * p
        if T > 4096:
            return None
        idx = [0] * (n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx[i * m + j, j * p + k, k * n + i] = T if False else (i * m + j) * (m * p) + (j * p + k) * (p * n) + (k * n + i)
        idx = np.array(idx, dtype=np.int64).reshape(n, m, p)
        # valid index for a term
        def valid(a, b, c):
            return (a < n * m) and (b < m * p) and (c < p * n)

        # enumerate candidate binary terms (a,b,c) with nonzero contribution
        cand = []
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if idx[a, b, c] < 0 or idx[a, b, c] >= T:
                        continue
                    cand.append((a, b, c))
        cand = np.array(cand, dtype=np.int64)
        if len(cand) == 0:
            return None
        cov = idx[cand[:, 0], cand[:, 1], cand[:, 2]]  # (C,)
        # remove duplicate coverage (keep first)
        _, first = np.unique(cov, return_index=True)
        keep = np.sort(first)
        cand = cand[keep]
        cov = cov[keep]
        C = len(cand)

        best = build_rank66()
        best_rank = 66

        max_rank = min(bound - 1, best_rank - 1)
        max_rank = max(max_rank, 1)

        for maxr in range(max_rank, 0, -1):
            if time.time() > deadline - 5:
                break
            # greedy set cover up to maxr terms
            chosen = []
            chosen_mask = np.zeros(T, dtype=np.bool_)
            avail = np.ones(C, dtype=np.bool_)
            for _ in range(maxr):
                if time.time() > deadline - 5:
                    break
                # coverage gain of each available term
                # cov is (C,); compute number of newly covered for each
                # vectorized: for each term, count uncovered in its cover
                # term covers a single index cov[t]
                # gain = 1 if not covered else 0
                gains = (~chosen_mask[cov]).astype(np.int64)
                gains[~avail] = -1
                t = int(np.argmax(gains))
                if gains[t] <= 0:
                    break
                chosen.append(t)
                chosen_mask[cov[t]] = True
                avail[t] = False
            if time.time() > deadline - 5:
                break
            if chosen_mask.all():
                r = len(chosen)
                if r < best_rank:
                    U = [[0] * r for _ in range(n * m)]
                    V = [[0] * r for _ in range(m * p)]
                    W = [[0] * r for _ in range(p * n)]
                    for q, t in enumerate(chosen):
                        a, b, c = cand[t]
                        U[a][q] = 1
                        V[b][q] = 1
                        W[c][q] = 1
                    if verify_decomp(U, V, W, r):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = r
                        return best
        return None

    # Guaranteed fallbacks
    fallback = build_rank66()
    if not verify_decomp(fallback["U"], fallback["V"], fallback["W"], len(fallback["U"][0])):
        fallback = build_schoolbook()
    best = fallback
    best_rank = len(fallback["U"][0])

    # Attempt to beat the block-Strassen bound
    if time.time() < deadline - 10:
        try:
            res = greedy_search(best_rank)
            if res is not None:
                r = len(res["U"][0])
                if r < best_rank:
                    best = res
                    best_rank = r
        except Exception:
            pass

    return best
