"""Combined program: clique-seeded greedy expansion + backtracking path routing."""
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _max_clique(n, adj):
    """Tomita's MCQ algorithm for maximum clique."""
    best = []
    best_size = 0
    deadline = time.time() + 80

    def color_sort(P_list):
        colors = [0] * len(P_list)
        order = list(range(len(P_list)))
        deg = [len(adj[P_list[i]] & set(P_list)) for i in range(len(P_list))]
        order.sort(key=lambda i: -deg[i])
        for idx in order:
            v = P_list[idx]
            used = set()
            for j in order:
                if j == idx:
                    break
                u = P_list[j]
                if u in adj[v] and colors[j] > 0:
                    used.add(colors[j])
            c = 1
            while c in used:
                c += 1
            colors[idx] = c
        order.sort(key=lambda i: colors[i])
        return order, colors

    def expand(clique, P_list):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if len(P_list) == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best = list(clique)
            return
        order, colors = color_sort(P_list)
        i = len(order) - 1
        while i >= 0:
            if time.time() > deadline:
                return
            v_idx = order[i]
            v = P_list[v_idx]
            if len(clique) + colors[v_idx] <= best_size:
                return
            clique.append(v)
            new_P = [P_list[order[j]] for j in range(i) if P_list[order[j]] in adj[v]]
            expand(clique, new_P)
            clique.pop()
            i -= 1

    expand([], list(range(n)))
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before the 90s wall-clock kill

    n = P['n']
    adj = _build_adj(n, P['edges'])
    adj_list = [sorted(adj[v]) for v in range(n)]

    def time_left():
        return deadline - time.time()

    def common_neighbors(u, v, branch_set, used_int):
        """Interior vertices adjacent to both u and v (not branch, not used)."""
        out = []
        for w in adj_list[u]:
            if w == v or w in branch_set or w in used_int:
                continue
            if w in adj[v]:
                out.append(w)
        return out

    def route(branches):
        """Find one simple path per pair with pairwise-disjoint interiors.
        Returns dict {(u,v): [u,...,v]} or None."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        direct = [p for p in pairs if p[1] in adj[p[0]]]
        need = [p for p in pairs if p[1] not in adj[p[0]]]

        # Feasibility: each non-direct pair needs a free common neighbor.
        for p in need:
            if not common_neighbors(p[0], p[1], branch_set, set()):
                return None

        # Order non-direct pairs by fewest options first (MRV).
        need.sort(key=lambda p: len(common_neighbors(p[0], p[1], branch_set, set())))

        used_int = set()
        assign = {}

        def bt(idx):
            if time_left() < 0.02:
                return False
            if idx == len(need):
                return True
            u, v = need[idx]
            for w in common_neighbors(u, v, branch_set, used_int):
                used_int.add(w)
                assign[(u, v)] = w
                if bt(idx + 1):
                    return True
                used_int.discard(w)
                del assign[(u, v)]
            return False

        if not bt(0):
            return None

        paths = {}
        for u, v in direct:
            paths[(u, v)] = [u, v]
        for (u, v), w in assign.items():
            paths[(u, v)] = [u, w, v]
        return paths

    def expand_greedy(cur_branches):
        """Try to add one more branch (highest-degree candidate first)."""
        cur_set = set(cur_branches)
        cands = [v for v in range(n) if v not in cur_set]
        cands.sort(key=lambda v: -len(adj[v]))
        for c in cands:
            trial = cur_branches + [c]
            r = route(trial)
            if r is not None:
                return trial, r
        return None

    # Baseline: maximum clique (always feasible with direct-edge paths).
    clique = _max_clique(n, adj)
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    best_paths = route(clique)
    best_k = len(clique)

    # Greedy expansion beyond the clique.
    cur = list(clique)
    while True:
        step = expand_greedy(cur)
        if step is None:
            break
        cur, best_paths = step
        best_k = len(cur)
        if time_left() < 0.5:
            break

    # Convert to exact Python types.
    branches = [int(x) for x in cur]
    paths_out = [[int(x) for x in best_paths[(u, v)]]
                 for (u, v) in combinations(branches, 2)]

    return {'branches': branches, 'paths': paths_out}
