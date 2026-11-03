"""Kissing number in dimension 11: support-level MIS with SA and 2-for-2 polish.

Structural change: operate on 4-element supports (330 vertices) instead of
individual vectors (5280). The conflict graph of weight-4 vectors (norm²=4,
four ±1 coordinates) factors through supports: two vectors conflict iff their
supports intersect in exactly 3. This association-scheme structure enables
fast vectorized operations and effective local search.
"""

import time
import numpy as np
import itertools
import random

D = 11


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Generate all C(11,4) = 330 supports
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330

    # Build conflict matrix: edge if intersection size == 3
    support_sets = [set(s) for s in supports]
    adj = np.zeros((n_sup, n_sup), dtype=np.uint8)
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            if len(support_sets[i] & support_sets[j]) == 3:
                adj[i, j] = 1
                adj[j, i] = 1

    # Precompute adjacency lists
    adj_lists = [np.where(adj[i] == 1)[0] for i in range(n_sup)]
    deg = adj.sum(axis=1)

    def greedy_is(seed=0):
        """Greedy independent set: repeatedly pick min-degree remaining vertex."""
        rng = np.random.default_rng(seed)
        remaining = np.ones(n_sup, dtype=np.uint8)
        selected = np.zeros(n_sup, dtype=np.uint8)

        while True:
            if time.time() > deadline - 10.0:
                break
            rem = np.where(remaining > 0)[0]
            if len(rem) == 0:
                break

            # Degree within remaining
            d_rem = np.array([len(adj_lists[v]) - len(np.intersect1d(adj_lists[v], np.where(remaining == 0)[0])) for v in rem])
            min_d = d_rem.min()
            cands = rem[d_rem == min_d]
            pick = rng.choice(cands)
            selected[pick] = 1
            remaining[pick] = 0
            remaining[adj[pick] == 1] = 0

        return selected

    def sa_search(S, n_steps, seed=0):
        """Simulated annealing with 1-for-k moves on support graph."""
        rng = np.random.default_rng(seed)
        rand = random.Random(seed)
        S = S.copy()
        cur = int(S.sum())
        best = cur
        best_S = S.copy()
        T = 1.5
        cool = 0.9998
        min_T = 0.03

        for step in range(n_steps):
            if time.time() > deadline - 8.0:
                break
            v = rng.integers(0, n_sup)

            if S[v]:
                # Probabilistic removal
                if rand.random() < 0.02:
                    S[v] = 0
                    cur -= 1
                continue

            # Try to add v
            conflicts = adj_lists[v][S[adj_lists[v]]]
            n_conf = len(conflicts)

            if n_conf == 0:
                S[v] = 1
                cur += 1
                if cur > best:
                    best = cur
                    best_S = S.copy()
            else:
                delta = 1 - n_conf
                if delta >= 0:
                    S[conflicts] = 0
                    S[v] = 1
                    cur += delta
                    if cur > best:
                        best = cur
                        best_S = S.copy()
                elif T > min_T:
                    if rand.random() < np.exp(delta / T):
                        S[conflicts] = 0
                        S[v] = 1
                        cur += delta
                        if cur > best:
                            best = cur
                            best_S = S.copy()
            T = max(min_T, T * cool)

        return best_S, best

    def polish_2for2(S, max_rounds=15):
        """2-for-2 swap: remove 2 selected supports, add 2 compatible ones."""
        rng = np.random.default_rng(12345)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5.0:
            improved = False
            rounds += 1
            sel = np.where(S == 1)[0]
            k = len(sel)
            if k < 2:
                break

            n_trials = min(3000, k * k // 2)
            for _ in range(n_trials):
                if time.time() > deadline - 5.0:
                    break
                i1, i2 = rng.integers(0, k, size=2)
                if i1 == i2:
                    continue
                v1, v2 = sel[i1], sel[i2]

                # Find supports not conflicting with v1 or v2, not selected
                compat = ((adj[v1] == 0) & (adj[v2] == 0) & (S == 0))
                cands = np.where(compat)[0]
                if len(cands) < 2:
                    continue

                # Try a few random pairs
                for _ in range(5):
                    u_idx = rng.integers(0, len(cands))
                    w_idx = rng.integers(0, len(cands))
                    if u_idx == w_idx:
                        continue
                    u, w = cands[u_idx], cands[w_idx]
                    if adj[u, w] == 1:
                        continue
                    # Check compatibility with other selected
                    others = np.delete(sel, [i1, i2])
                    if len(others) > 0:
                        if np.any(adj[u][others]) or np.any(adj[w][others]):
                            continue
                    S[v1] = 0
                    S[v2] = 0
                    S[u] = 1
                    S[w] = 1
                    improved = True
                    break
                if improved:
                    break
        return S

    def expand_to_vectors(S):
        """Expand support selection to actual vectors, one per support.
        Pick the sign pattern that maximizes separation from other chosen vectors."""
        vectors = []
        sel_supports = np.where(S == 1)[0]
        for idx in sel_supports:
            sup = supports[idx]
            # Try all 16 sign patterns, pick one with max min-distance to existing
            best_signs = None
            best_min_dist = -1
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for c, s in zip(sup, signs):
                    v[c] = s
                v = np.array(v)
                if not vectors:
                    best_signs = v
                    best_min_dist = 999
                else:
                    min_d = min(np.sum((v - u) ** 2) for u in vectors)
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_signs = v
            vectors.append(best_signs)
        return vectors

    # Main search: multiple restarts with greedy + SA + polish
    best_size = 0
    best_S = np.zeros(n_sup, dtype=np.uint8)

    n_restarts = 40
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        S_init = greedy_is(seed=1000 + i * 7)
        init_size = int(S_init.sum())

        # SA refinement
        n_steps = 40000 if i % 4 == 0 else 25000
        S_sa, sa_size = sa_search(S_init, n_steps, seed=5000 + i * 13)

        # Polish
        S_pol = polish_2for2(S_sa.copy(), max_rounds=12)
        pol_size = int(S_pol.sum())

        if pol_size > best_size:
            best_size = pol_size
            best_S = S_pol.copy()

    # Final long SA run on best solution
    if time.time() < deadline - 20.0:
        S_final, fin_size = sa_search(best_S, 60000, seed=99999)
        S_final = polish_2for2(S_final, max_rounds=10)
        if int(S_final.sum()) > best_size:
            best_size = int(S_final.sum())
            best_S = S_final

    # Expand to vectors
    vectors = expand_to_vectors(best_S)

    if not vectors:
        # Fallback
        vectors = [[0] * D for _ in range(D)]
        for i in range(D):
            vectors[i][i] = 1

    return np.array(vectors, dtype=np.int64)
