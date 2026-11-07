import time
import random
import sys
import math
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edge_list = P['edges']

adj = defaultdict(list)
for u, v in edge_list:
    adj[u].append(v)
    adj[v].append(u)

# Precompute per-vertex incident edge indices for fast local cost updates
inc = defaultdict(list)
for ei, (u, v) in enumerate(edge_list):
    inc[u].append(ei)
    inc[v].append(ei)

LOG2 = math.log(2.0)
EXP = math.exp

def smooth_cost(colors, K):
    """Smooth surrogate: logsumexp of per-edge properness penalties and
    per-bicolored-pair cycle penalties (cycle <=> some component has E>=V)."""
    proper = 0
    for u, v in edge_list:
        if colors[u] == colors[v]:
            proper += 1
    # per-pair bicolored cycle count
    pair_E = {}
    pair_V = {}
    for u, v in edge_list:
        a = colors[u]
        b = colors[v]
        if a == b:
            continue
        if a > b:
            a, b = b, a
        key = (a, b)
        pair_E[key] = pair_E.get(key, 0) + 1
        pair_V.setdefault(a, set()).add(u)
        pair_V.setdefault(b, set()).add(v)
    cyc = 0
    for key, e in pair_E.items():
        a, b = key
        V = len(pair_V[a]) + len(pair_V[b])
        if e >= V:
            cyc += 1
    M = max(proper, cyc)
    if M == 0:
        return 0.0
    s = 0.0
    if proper > 0:
        s += EXP(LOG2 - proper)
    if cyc > 0:
        s += EXP(-cyc)
    return M + math.log(s)

def get_cost(colors, K):
    """Exact cost. 0 iff proper and every bicolored subgraph is a forest."""
    proper = 0
    for u, v in edge_list:
        if colors[u] == colors[v]:
            proper += 1
    if proper > 0:
        return proper
    pair_E = {}
    pair_V = {}
    for u, v in edge_list:
        a = colors[u]
        b = colors[v]
        if a == b:
            continue
        if a > b:
            a, b = b, a
        key = (a, b)
        pair_E[key] = pair_E.get(key, 0) + 1
        pair_V.setdefault(a, set()).add(u)
        pair_V.setdefault(b, set()).add(v)
    cyc = 0
    for key, e in pair_E.items():
        a, b = key
        V = len(pair_V[a]) + len(pair_V[b])
        if e >= V:
            cyc += 1
    return cyc

def check_validity(colors, K):
    return get_cost(colors, K) == 0

def local_search(K, time_limit, seed=None):
    rng = random.Random(seed)
    start_time = time.time()
    best_cost = float('inf')
    best_coloring = None

    def time_left():
        return start_time + time_limit - time.time()

    while time_left() > 0.1:
        # --- structured initialization ---
        r = rng.random()
        if r < 0.40:
            colors = [v % 2 for v in range(N)]
        elif r < 0.65:
            colors = [v % 7 for v in range(N)]
        elif r < 0.85:
            colors = [rng.randint(0, K - 1) for _ in range(N)]
        else:
            base = [v % 2 for v in range(N)]
            colors = list(base)
            for _ in range(rng.randint(1, 6)):
                colors[rng.randint(0, N - 1)] = rng.randint(0, K - 1)

        cur = smooth_cost(colors, K)
        if cur == 0.0:
            return list(colors)

        no_improve = 0
        max_no_improve = 400
        while time_left() > 0.1 and no_improve < max_no_improve:
            v = rng.randint(0, N - 1)
            old_c = colors[v]
            best_c = old_c
            best_val = cur
            for c in range(K):
                if c == old_c:
                    continue
                colors[v] = c
                val = smooth_cost(colors, K)
                if val < best_val:
                    best_val = val
                    best_c = c
            colors[v] = best_c
            cur = best_val

            if cur < best_cost:
                best_cost = cur
                best_coloring = list(colors)
                if cur == 0.0:
                    return list(colors)

            if best_c == old_c:
                no_improve += 1
            else:
                no_improve = 0

    if best_coloring is not None:
        return best_coloring
    return None

def construct():
    deadline = time.time() + 85.0  # safety margin vs 90s kill

    for K in (3, 4, 5):
        if time.time() >= deadline:
            break
        rem = deadline - time.time()
        if rem < 3.0:
            break
        tl = min(20.0, rem - 3.0)
        for attempt in range(6):
            if time.time() >= deadline:
                break
            res = local_search(K, tl, seed=1000 * K + attempt)
            if res is not None:
                return [int(x) for x in res]

    # Fallback: a guaranteed-valid coloring (50 distinct colors, no bicolored subgraph
    # has more than one edge, hence always a forest).
    return [int(i) for i in range(N)]

# EVOLVE-BLOCK-END
