"""Rewritten program for Simple Assembly Line Balancing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def _build_graph(n, preds):
    succs = [[] for _ in range(n)]
    for i, ps in enumerate(preds):
        for p in ps:
            succs[p].append(i)
    indeg = np.zeros(n, dtype=int)
    for i, ps in enumerate(preds):
        for p in ps:
            indeg[i] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:
        seen = set(order)
        order.extend([i for i in range(n) if i not in seen])
    return order, succs

def _simulate(times, preds, succs, ct, perm):
    n = len(times)
    unassigned_preds = [len(preds[i]) for i in range(n)]
    assigned = [False] * n
    stations = 0
    cur = 0.0
    remaining = n
    while remaining > 0:
        best = -1
        best_rank = -1
        for r, i in enumerate(perm):
            if not assigned[i] and unassigned_preds[i] == 0:
                if times[i] <= ct - cur:
                    best = i
                    best_rank = r
                    break
        if best == -1:
            if cur > 0:
                stations += 1
                cur = 0.0
                continue
            else:
                break
        assigned[best] = True
        cur += times[best]
        remaining -= 1
        for s in succs[best]:
            unassigned_preds[s] -= 1
    if cur > 0:
        stations += 1
    return stations

def _norm(x):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def priority(instance):
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    ct = float(instance["cycle_time"])

    order, succs = _build_graph(n, preds)
    rev_order = list(reversed(order))

    # --- Feature 1: Critical path (longest downstream path including own time) ---
    cp = np.zeros(n, dtype=float)
    for i in rev_order:
        best = 0.0
        for s in succs[i]:
            if cp[s] > best:
                best = cp[s]
        cp[i] = times[i] + best

    # --- Feature 2: Transitive successor count ---
    tsucc = np.zeros(n, dtype=int)
    for i in rev_order:
        c = 0
        for s in succs[i]:
            c += 1 + tsucc[s]
        tsucc[i] = c

    # --- Feature 3: Flow-capacity criticality (directive: reciprocal of downstream max-flow capacity) ---
    # Max flow from task i to sink in unit-capacity DAG = min-cut = min number of tasks in any
    # antichain separating i from sink. Approximate by min-cut over unit capacities via max-flow.
    # For n=30 this is cheap. We compute max flow from i to a super-sink.
    def _maxflow_unit(src):
        # Build adjacency with capacities = 1 per edge, add super-sink T
        # Nodes 0..n-1, sink = n
        T = n
        # capacity matrix
        cap = np.zeros((n + 1, n + 1), dtype=int)
        for u in range(n):
            for v in succs[u]:
                cap[u, v] = 1
            if not succs[u]:
                cap[u, T] = 1  # direct to sink
        # Edmonds-Karp (BFS augmenting paths)
        flow = 0
        while True:
            parent = [-1] * (n + 1)
            parent[src] = -2
            q = deque([src])
            found = False
            while q and not found:
                u = q.popleft()
                for v in range(n + 1):
                    if parent[v] == -1 and cap[u, v] > 0:
                        parent[v] = u
                        if v == T:
                            found = True
                            break
                        q.append(v)
            if not found:
                break
            # find bottleneck
            addf = 10**9
            v = T
            while v != src:
                u = parent[v]
                addf = min(addf, cap[u, v])
                v = u
            v = T
            while v != src:
                u = parent[v]
                cap[u, v] -= addf
                cap[v, u] += addf
                v = u
            flow += addf
        return flow

    flow_cap = np.zeros(n, dtype=float)
    for i in range(n):
        flow_cap[i] = float(_maxflow_unit(i))
    # reciprocal: smaller flow capacity -> higher criticality
    flow_crit = 1.0 / (flow_cap + 1e-9)

    # --- Feature 4: Topological position index (earlier in topo = more upstream) ---
    pos_idx = np.zeros(n, dtype=float)
    for r, i in enumerate(order):
        pos_idx[i] = r

    # --- Build candidate priority vectors (each gives a permutation via argsort) ---
    cp_n = _norm(cp)
    time_n = _norm(times)
    tsucc_n = _norm(tsucc.astype(float))
    flow_n = _norm(flow_crit)
    pos_n = _norm(pos_idx)

    candidates = [
        0.85 * cp_n + 0.15 * time_n,
        0.7 * cp_n + 0.3 * time_n,
        0.6 * cp_n + 0.25 * time_n + 0.15 * flow_n,
        0.5 * cp_n + 0.3 * time_n + 0.2 * flow_n,
        0.4 * cp_n + 0.4 * time_n + 0.2 * flow_n,
        0.3 * cp_n + 0.5 * time_n + 0.2 * flow_n,
        0.5 * cp_n + 0.3 * time_n + 0.2 * tsucc_n,
        0.4 * cp_n + 0.3 * time_n + 0.2 * flow_n + 0.1 * tsucc_n,
        0.3 * cp_n + 0.4 * time_n + 0.3 * flow_n,
        0.2 * cp_n + 0.5 * time_n + 0.3 * flow_n,
        0.5 * cp_n + 0.2 * time_n + 0.3 * pos_n,
        0.4 * cp_n + 0.3 * time_n + 0.2 * pos_n + 0.1 * flow_n,
        time_n,
        cp_n,
        flow_n,
    ]

    best_perm = None
    best_stations = float('inf')

    for cand in candidates:
        perm = np.argsort(-cand, kind='stable')  # descending priority
        s = _simulate(times, preds, succs, ct, perm)
        if s < best_stations:
            best_stations = s
            best_perm = perm.copy()

    # --- Local search: swap-based permutation search ---
    # A swap of two positions in the permutation changes the priority ordering.
    # We accept if station count decreases; if equal, we keep (to maintain diversity).
    # Use a randomized schedule of swaps, biased toward adjacent and critical pairs.

    rng = np.random.default_rng(42)
    no_improve = 0
    max_no_improve = 30000
    step_count = 0

    while time.time() - t0 < deadline:
        if no_improve > max_no_improve:
            break

        # Choose a swap: with prob 0.5, swap adjacent; else random pair
        if rng.random() < 0.5:
            # adjacent swap in the permutation
            a = int(rng.integers(0, n - 1))
            b = a + 1
        else:
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            if a == b:
                a = (a + 1) % n

        # Apply swap
        best_perm[a], best_perm[b] = best_perm[b], best_perm[a]
        step_count += 1
        s = _simulate(times, preds, succs, ct, best_perm)

        if s < best_stations:
            best_stations = s
            no_improve = 0
        else:
            # revert
            best_perm[a], best_perm[b] = best_perm[b], best_perm[a]
            no_improve += 1

        # Periodically try a "reheat": swap a random pair of non-adjacent elements
        # to escape local minima (already covered by the random-pair branch above)

    # Convert best permutation back to a priority score vector
    # The evaluator assigns higher priority to higher score.
    # In our simulation, perm[0] is the highest-priority task.
    # So score[perm[i]] should be decreasing in i.
    scores = np.zeros(n, dtype=float)
    for rank, task in enumerate(best_perm):
        scores[task] = float(n - rank)  # rank 0 -> highest score

    return scores

# EVOLVE-BLOCK-END
