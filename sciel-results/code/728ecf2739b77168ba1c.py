import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 143

    # Generate candidates: all vectors with 4 nonzero entries each being ±1, or one entry ±2
    candidates = []

    # Type 1: ±2e_i (Norm sq 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: ±e_i ± e_j ± e_k ± e_l (Norm sq 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix
    M = candidates @ candidates.T

    # All have norm sq 4. Distance sq = 8 - 2*dot. Need >= 4, so dot <= 2.
    # Conflict if dot > 2, i.e., dot >= 3 (integer)
    conflict = M > 2

    # Build conflict adjacency
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if conflict[i, j]:
                adj[i].add(j)
                adj[j].add(i)

    # Build compatibility adjacency (for clique formulation)
    compat_adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if not conflict[i, j]:
                compat_adj[i].add(j)
                compat_adj[j].add(i)

    best_set = []
    best_count = 0

    def time_check():
        return time.time() > deadline

    # Maximum clique via Tomita-style branch and bound
    def expand(P, R):
        nonlocal best_set, best_count

        if time_check():
            return

        if len(R) + len(P) <= best_count:
            return

        if not P:
            if len(R) > best_count:
                best_count = len(R)
                best_set = R[:]
            return

        # Greedy coloring for upper bound
        # Sort by degree descending
        P_set = set(P)
        P_sorted = sorted(P, key=lambda x: -len(compat_adj[x] & P_set))

        colors = [0] * len(P_sorted)
        color_count = 0

        for i, v in enumerate(P_sorted):
            used = set()
            for j in range(i):
                if P_sorted[j] in compat_adj[v]:
                    used.add(colors[j])

            c = 0
            while c in used:
                c += 1
            colors[i] = c
            color_count = max(color_count, c + 1)

        # Process in reverse color order
        for i in range(len(P_sorted) - 1, -1, -1):
            if time_check():
                return

            v = P_sorted[i]
            c = colors[i] + 1  # 1-indexed upper bound

            if len(R) + c <= best_count:
                continue

            new_R = R + [v]
            if len(new_R) > best_count:
                best_count = len(new_R)
                best_set = new_R[:]

            new_P = [P_sorted[j] for j in range(i) if P_sorted[j] in compat_adj[v]]
            expand(new_P, new_R)

    # Run maximum clique
    if time.time() < deadline - 5:
        P_initial = list(range(N))
        expand(P_initial, [])

    # Fallback greedy if needed
    if best_count == 0 or time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        conflict_counts = np.array([len(adj[i]) for i in range(N)])

        for trial in range(30):
            if time_check():
                break

            if trial == 0:
                order = np.argsort(conflict_counts, kind='stable')
            else:
                order = rng.permutation(N)

            selected = []
            sel_set = set()

            for idx in order:
                if not any(idx in adj[s] for s in selected):
                    selected.append(idx)
                    sel_set.add(idx)

            if len(selected) > best_count:
                best_count = len(selected)
                best_set = selected[:]

    # Local improvement
    if best_set and time.time() < deadline - 2:
        sel_set = set(best_set)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for idx in range(N):
                if idx in sel_set:
                    continue
                if not any(idx in adj[s] for s in sel_set):
                    sel_set.add(idx)
                    best_set.append(idx)
                    best_count += 1
                    improved = True

    if not best_set:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return candidates[best_set].astype(np.int64)
