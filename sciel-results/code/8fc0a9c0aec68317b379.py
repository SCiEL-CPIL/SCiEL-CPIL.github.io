import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            S = {u, v} | adj[u] | adj[v]
            for j, (x, y) in enumerate(edges):
                if i == j:
                    continue
                if x in S or y in S:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        return conflicts

    def max_induced_matching(conflicts, m, rng):
        """Greedy maximum induced matching (randomized tie-breaking)."""
        if m == 0:
            return []
        used = [False] * m
        order = list(range(m))
        rng.shuffle(order)
        order.sort(key=lambda i: len(conflicts[i]))  # fewest conflicts first (less restrictive)
        chosen = []
        for i in order:
            if used[i]:
                continue
            # check no conflict with chosen
            ok = True
            for j in chosen:
                if j in conflicts[i]:
                    ok = False
                    break
            if not ok:
                continue
            chosen.append(i)
            used[i] = True
            for nb in conflicts[i]:
                used[nb] = True
        return chosen

    def dsatur_coloring(conflicts, m, rng):
        if m == 0:
            return [], 0
        color = [-1] * m
        neighbour_colors = [set() for _ in range(m)]
        degrees = [len(conflicts[i]) for i in range(m)]
        uncoloured = list(range(m))
        while uncoloured:
            best, best_sat, best_deg = None, -1, -1
            for v in uncoloured:
                s = len(neighbour_colors[v])
                if s > best_sat or (s == best_sat and degrees[v] > best_deg):
                    best, best_sat, best_deg = v, s, degrees[v]
            uncoloured.remove(best)
            used = neighbour_colors[best]
            c = 0
            while c in used:
                c += 1
            color[best] = c
            for nb in conflicts[best]:
                if color[nb] == -1:
                    neighbour_colors[nb].add(c)
        return color, (max(color) + 1)

    def greedy_color(conflicts, m, rng, strategy):
        if m == 0:
            return [], 0
        color_of_edge = [-1] * m
        if strategy == 'largest_first':
            order = sorted(range(m), key=lambda i: len(conflicts[i]), reverse=True)
        elif strategy == 'random':
            order = list(range(m))
            rng.shuffle(order)
        elif strategy == 'smallest_first':
            order = sorted(range(m), key=lambda i: len(conflicts[i]))
        else:
            rvals = [rng.random() for _ in range(m)]
            order = sorted(range(m), key=lambda i: (-len(conflicts[i]), rvals[i]))
        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if color_of_edge[j] != -1:
                    forbidden.add(color_of_edge[j])
            c = 0
            while c in forbidden:
                c += 1
            color_of_edge[i] = c
        return color_of_edge, (max(color_of_edge) + 1)

    def local_search(conflicts, m, colors, rng, max_iters=200):
        """Local search with reassignment + 2-swap moves to reduce color count."""
        if m == 0:
            return colors, 0
        cur = colors[:]
        nc = max(cur) + 1
        for _ in range(max_iters):
            improved = False
            # Phase 1: reassign to lower color
            for v in range(m):
                cc = cur[v]
                for c in range(cc):
                    ok = True
                    for nb in conflicts[v]:
                        if cur[nb] == c:
                            ok = False
                            break
                    if ok:
                        cur[v] = c
                        improved = True
                        break
            # Phase 2: try to eliminate highest color
            if nc > 1:
                top = nc - 1
                top_verts = [v for v in range(m) if cur[v] == top]
                for v in top_verts:
                    for c in range(top):
                        ok = True
                        for nb in conflicts[v]:
                            if cur[nb] == c:
                                ok = False
                                break
                        if ok:
                            cur[v] = c
                            improved = True
                            break
            # Phase 3: 2-swap moves
            for v in range(m):
                for w in range(v + 1, m):
                    if cur[v] == cur[w]:
                        continue
                    cv, cw = cur[v], cur[w]
                    # try swap
                    ok = True
                    for nb in conflicts[v]:
                        if nb != w and cur[nb] == cw:
                            ok = False
                            break
                    if ok:
                        for nb in conflicts[w]:
                            if nb != v and cur[nb] == cv:
                                ok = False
                                break
                    if ok:
                        cur[v], cur[w] = cw, cv
                        improved = True
                        break
            if not improved:
                break
            new_max = max(cur) + 1
            if new_max < nc:
                nc = new_max
        return cur, (max(cur) + 1)

    def color_graph(edges, adj, conflicts, m, rng):
        if m == 0:
            return [], 0
        best_colors, best_score = None, float('inf')

        # DSATUR
        c, s = dsatur_coloring(conflicts, m, rng)
        if s < best_score:
            best_score, best_colors = s, c
        # Greedy strategies
        for strat in ('largest_first', 'random', 'smallest_first', 'largest_random_tie'):
            c, s = greedy_color(conflicts, m, rng, strat)
            if s < best_score:
                best_score, best_colors = s, c
        # Local search on best
        c, s = local_search(conflicts, m, best_colors, rng, max_iters=150)
        if s < best_score:
            best_score, best_colors = s, c
        return best_colors, best_score

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    graphs = []
    for n, d, seed in specs:
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        conflicts = build_conflict_graph(edges, adj)
        graphs.append((edges, adj, conflicts, len(edges)))

    best_results = [None] * len(graphs)
    best_scores = [float('inf')] * len(graphs)

    # Initial solve
    for idx, (edges, adj, conflicts, m) in enumerate(graphs):
        rng = random.Random(0)
        c, s = color_graph(edges, adj, conflicts, m, rng)
        best_results[idx], best_scores[idx] = c, s

    # Multi-restart with time budget
    iter_no = 0
    while time.time() < deadline - 3.0:
        iter_no += 1
        rng = random.Random(10000 + iter_no)
        for idx, (edges, adj, conflicts, m) in enumerate(graphs):
            if time.time() > deadline - 2.0:
                break
            c, s = color_graph(edges, adj, conflicts, m, rng)
            if s < best_scores[idx]:
                best_scores[idx] = s
                best_results[idx] = c

    # Final polish on each best
    for idx, (edges, adj, conflicts, m) in enumerate(graphs):
        if best_results[idx] is not None:
            c, s = local_search(conflicts, m, best_results[idx], random.Random(777), max_iters=300)
            if s < best_scores[idx]:
                best_scores[idx] = s
                best_results[idx] = c

    for i in range(len(graphs)):
        if best_results[i] is None:
            edges, adj, conflicts, m = graphs[i]
            c, _ = color_graph(edges, adj, conflicts, m, random.Random(0))
            best_results[i] = c

    return best_results
