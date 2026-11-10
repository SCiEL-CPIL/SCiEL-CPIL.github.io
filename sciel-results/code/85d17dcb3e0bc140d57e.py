import time
import numpy as np
from itertools import combinations


def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before the 150s kill

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_pairs)
    u_arr = np.array([p[0] for p in all_pairs], dtype=np.int64)
    v_arr = np.array([p[1] for p in all_pairs], dtype=np.int64)

    # Precompute, for each pair (edge index), the set of triples that contain it.
    # A K3,3 is witnessed by a triple with >= 3 common neighbours; any triple
    # containing an edge (u,v) can only be affected when that edge is toggled.
    triples = list(combinations(range(n), 3))
    triples_arr = np.array(triples, dtype=np.int64)            # (T,3)
    T = len(triples)

    # pair_of_triplet: for each triple and each of its 3 slots, the edge index
    # of the pair formed by the other two vertices.
    pair_of_triplet = np.zeros((T, 3), dtype=np.int64)
    for t in range(T):
        a, b, c = triples[t]
        pair_of_triplet[t, 0] = all_pairs.index((b, c))
        pair_of_triplet[t, 1] = all_pairs.index((a, c))
        pair_of_triplet[t, 2] = all_pairs.index((a, b))

    # pair_to_triples: list of triple indices containing each edge.
    pair_to_triples = [[] for _ in range(m)]
    for t in range(T):
        for s in range(3):
            pair_to_triples[pair_of_triplet[t, s]].append(t)
    pair_to_triples = [np.array(lst, dtype=np.int64) for lst in pair_to_triples]

    # Penalty per triple with k common neighbours (k>=3).  Monotonically
    # increasing and large enough that any K3,3 is never worth keeping.
    PEN = np.zeros(1, dtype=np.float64)
    PEN[1:] = np.array([0.0, 0.0, 1.0, 4.0, 10.0, 22.0, 40.0, 70.0,
                        110.0, 160.0, 220.0, 300.0, 400.0, 520.0, 660.0,
                        820.0, 1000.0, 1200.0, 1420.0, 1660.0, 1920.0],
                       dtype=np.float64)

    def compute_commons(e):
        """Return int array of length T: common-neighbor count per triple."""
        cm = (e[:, u_arr] & e[:, v_arr]).astype(np.int64)
        return cm.sum(axis=1)

    def penalty(com):
        return PEN[com].sum()

    def edge_count(e):
        return int(e.sum())

    def is_k33_free(e):
        com = compute_commons(e)
        return bool(com.max() < 3)

    def build_e():
        e = np.zeros(m, dtype=np.uint8)
        for idx, (i, j) in enumerate(all_pairs):
            e[idx] = 1
        return e

    # ---- baseline: complete graph minus one edge (guaranteed K3,3-free) ----
    e_base = build_e()
    e_base[m - 1] = 0
    com_base = compute_commons(e_base)
    base_score = edge_count(e_base) - penalty(com_base)
    best_e = e_base.copy()
    best_score = base_score
    best_count = edge_count(best_e)

    rng = np.random.default_rng(12345)

    def run_anneal(seed, time_budget):
        nonlocal best_e, best_score, best_count
        rng = np.random.default_rng(seed)

        # random dense start
        e = (rng.random(m) < 0.5).astype(np.uint8)
        com = compute_commons(e)
        score = edge_count(e) - penalty(com)

        cur_e = e
        cur_com = com
        cur_score = score

        best_e_local = cur_e.copy()
        best_score_local = cur_score

        T0 = 12.0
        T1 = 0.15
        end = time.time() + time_budget
        steps = 0
        while time.time() < end:
            # pick a batch of candidate edges
            B = 48
            idx = rng.integers(0, m, size=B)
            e_new = cur_e.copy()
            e_new[idx] = 1 - e_new[idx]
            com_new = compute_commons(e_new)
            score_new = edge_count(e_new) - penalty(com_new)

            # temperature schedule
            T = T0 * (T1 / T0) ** (steps / 4000.0)
            if T < T1:
                T = T1
            steps += 1

            # Metropolis accept the batch (vectorized decision)
            d = score_new - cur_score
            accept = (d >= 0) | (rng.random(B) < np.exp(d / T))
            if accept.any():
                cur_e = e_new
                cur_com = com_new
                cur_score = score_new

            # track best by raw edge count among K3,3-free states
            if cur_com.max() < 3:
                c = edge_count(cur_e)
                if c > edge_count(best_e_local):
                    best_e_local = cur_e.copy()
                    best_score_local = cur_score

            # occasional single-edge fine-tune near the end (low T)
            if T < 3.0 and rng.random() < 0.5:
                i = int(rng.integers(0, m))
                e1 = cur_e.copy()
                e1[i] = 1 - e1[i]
                com1 = compute_commons(e1)
                s1 = edge_count(e1) - penalty(com1)
                d1 = s1 - cur_score
                if d1 >= 0 or rng.random() < np.exp(d1 / T):
                    cur_e = e1
                    cur_com = com1
                    cur_score = s1
                    if cur_com.max() < 3 and edge_count(cur_e) > edge_count(best_e_local):
                        best_e_local = cur_e.copy()
                        best_score_local = cur_score

        # final greedy completion: add any edge that keeps it K3,3-free
        if cur_com.max() < 3:
            for i in range(m):
                if cur_e[i] == 0:
                    e1 = cur_e.copy()
                    e1[i] = 1
                    com1 = compute_commons(e1)
                    if com1.max() < 3:
                        cur_e = e1
                        cur_com = com1
            if cur_com.max() < 3 and edge_count(cur_e) > edge_count(best_e_local):
                best_e_local = cur_e.copy()
                best_score_local = cur_score

        # keep global best (prefer more edges; tie-break by lower penalty)
        if edge_count(best_e_local) > best_count or (
                edge_count(best_e_local) == best_count and
                best_score_local > best_score):
            best_e = best_e_local.copy()
            best_score = best_score_local
            best_count = edge_count(best_e)

    # Run multiple annealing phases with decreasing time budgets.
    while time.time() < deadline - 8:
        remaining = deadline - time.time()
        if remaining > 60:
            budget = 30.0
        elif remaining > 25:
            budget = 15.0
        elif remaining > 12:
            budget = 8.0
        else:
            budget = 5.0
        run_anneal(1000 + int(time.time() * 1000) % 100000, budget)

    # Guarantee a valid K3,3-free result.
    if not is_k33_free(best_e):
        # repair: drop edges until free
        e = best_e.copy()
        com = compute_commons(e)
        guard = 0
        while com.max() >= 3 and guard < 2000:
            guard += 1
            worst = int(np.argmax(com))
            # find an edge in this triple to remove that reduces the count most
            a, b, c = triples[worst]
            cands = [all_pairs.index((b, c)), all_pairs.index((a, c)),
                     all_pairs.index((a, b))]
            best_i, best_com = None, None
            for i in cands:
                e1 = e.copy()
                e1[i] = 0
                com1 = compute_commons(e1)
                if best_com is None or com1.sum() < best_com.sum():
                    best_com = com1
                    best_i = i
            e = e.copy()
            e[best_i] = 0
            com = compute_commons(e)
        best_e = e
        best_count = edge_count(best_e)

    # Final polish: greedy add any remaining feasible edge.
    com = compute_commons(best_e)
    for i in range(m):
        if best_e[i] == 0:
            e1 = best_e.copy()
            e1[i] = 1
            if compute_commons(e1).max() < 3:
                best_e = e1
                best_count = edge_count(best_e)

    edges = [list(all_pairs[i]) for i in range(m) if best_e[i] == 1]
    return edges


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    # quick validity check
    n = 20
    adj = [[0] * n for _ in range(n)]
    for u, v in res:
        adj[u][v] = 1
        adj[v][u] = 1
    ok = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                common = 0
                for x in range(n):
                    if adj[i][x] and adj[j][x] and adj[k][x]:
                        common += 1
                if common >= 3:
                    ok = False
                    print("K3,3 at", i, j, k, "common", common)
    print("K3,3-free:", ok)
