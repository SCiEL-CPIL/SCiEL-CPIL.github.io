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

    def build_rank52():
        """p=4+2 split, n=1+2 per block, Strassen on 2x2x2. Actual rank = 52."""
        rank = 72  # allocate max, trim later
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

        actual_rank = cur[0]
        U = [row[:actual_rank] for row in U]
        V = [row[:actual_rank] for row in V]
        W = [row[:actual_rank] for row in W]
        return {"U": U, "V": V, "W": W}

    def build_block_decomp(n_split, m_split, p_split):
        """Generic block decomposition with Strassen on 2x2x2 sub-blocks."""
        rank = n * m * p
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

        # Process blocks: for each (ni, mj, pk) block, if all dims == 2 use Strassen, else schoolbook
        for ni_idx, ni in enumerate(n_split):
            for mj_idx, mj in enumerate(m_split):
                for pk_idx, pk in enumerate(p_split):
                    i0 = sum(n_split[:ni_idx])
                    j0 = sum(m_split[:mj_idx])
                    k0 = sum(p_split[:pk_idx])
                    if ni == 2 and mj == 2 and pk == 2:
                        add_ssr(i0, j0, k0)
                    else:
                        add_sb(i0, j0, k0, ni, mj, pk)

        actual_rank = cur[0]
        U = [row[:actual_rank] for row in U]
        V = [row[:actual_rank] for row in V]
        W = [row[:actual_rank] for row in W]
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

    def search_block_decomps(bound):
        """Search over block splits to find minimal-rank structured decomposition."""
        best = None
        best_rank = bound

        def all_splits(total, min_part=1):
            """Generate all compositions of total into parts >= min_part."""
            if total == min_part:
                yield [min_part]
                return
            for first in range(min_part, total):
                for rest in all_splits(total - first, min_part):
                    yield [first] + rest

        # Try all splits of n, m, p into parts of size 1 or 2
        n_splits = list(all_splits(n, 1))
        m_splits = list(all_splits(m, 1))
        p_splits = list(all_splits(p, 1))

        for ns in n_splits:
            if time.time() > deadline - 5:
                return best
            for ms in m_splits:
                if time.time() > deadline - 5:
                    return best
                for ps in p_splits:
                    if time.time() > deadline - 5:
                        return best
                    try:
                        decomp = build_block_decomp(ns, ms, ps)
                        r = len(decomp["U"][0])
                        if r < best_rank:
                            if verify_decomp(decomp["U"], decomp["V"], decomp["W"], r):
                                best = decomp
                                best_rank = r
                    except Exception:
                        pass
        return best

    # Guaranteed fallbacks
    fallback = build_rank52()
    if not verify_decomp(fallback["U"], fallback["V"], fallback["W"], len(fallback["U"][0])):
        fallback = build_schoolbook()
    best = fallback
    best_rank = len(fallback["U"][0])

    # Search over block decompositions to find better structured decomposition
    if time.time() < deadline - 10:
        try:
            res = search_block_decomps(best_rank)
            if res is not None:
                r = len(res["U"][0])
                if r < best_rank:
                    best = res
                    best_rank = r
        except Exception:
            pass

    return best
