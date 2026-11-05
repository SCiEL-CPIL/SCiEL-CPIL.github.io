import time
import random
from collections import defaultdict
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    def make_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(6 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def build_conflicts_np(edges, adj, n):
        """Vectorized conflict graph via NumPy. Two edges conflict if they share
        a vertex or are joined by an edge (i.e., a vertex of one is adjacent to
        a vertex of the other)."""
        m = len(edges)
        if m == 0:
            return []
        # endpoints as int array
        U = np.array([e[0] for e in edges], dtype=np.int32)
        V = np.array([e[1] for e in edges], dtype=np.int32)
        # For each edge i, the "forbidden vertex set" S_i = {u_i, v_i} ∪ adj[u_i] ∪ adj[v_i]
        # Build a matrix S of shape (m, n) boolean: S[i, x] = 1 if x in S_i
        S = np.zeros((m, n), dtype=bool)
        S[np.arange(m), U] = True
        S[np.arange(m), V] = True
        # For each vertex x, find which edges have x in their endpoint set
        # and mark all neighbours of x as in S_i for those edges.
        # For each vertex x:
        #   edges_incident_to_x = indices i where U[i]==x or V[i]==x
        #   for each y in adj[x]: S[edges_incident_to_x, y] = True
        for x in range(n):
            idx = np.where((U == x) | (V == x))[0]
            if len(idx) == 0:
                continue
            for y in adj[x]:
                S[idx, y] = True
        # Conflict matrix: C[i,j] = True iff (U[i] in S_j) or (V[i] in S_j)
        # i.e., S @ E^T where E has 1s at (i, U[i]) and (i, V[i])
        E = np.zeros((m, n), dtype=bool)
        E[np.arange(m), U] = True
        E[np.arange(m), V] = True
        # C[i,j] = OR over x of (S[j,x] and E[i,x])
        # Use matrix multiplication with int8 then threshold
        S_int = S.astype(np.int8)
        E_int = E.astype(np.int8)
        C = S_int @ E_int.T  # shape (m, m), C[i,j] = sum_x S[j,x]*E[i,x]
        C_bool = (C > 0)
        np.fill_diagonal(C_bool, False)
        # Convert to list of sets for fast local search
        conflicts = []
        for i in range(m):
            row = C_bool[i]
            idx = np.where(row)[0]
            conflicts.append(set(int(x) for x in idx))
        return conflicts

    def dsatur_coloring(conflicts, m, time_budget=0.0):
        """DSATUR heuristic. Returns (color_array, num_colors)."""
        if m == 0:
            return [], 0
        color = [-1] * m
        neighbour_colors = [set() for _ in range(m)]
        degrees = [len(conflicts[i]) for i in range(m)]
        uncoloured = list(range(m))
        t0 = time.time()
        while uncoloured:
            if time_budget > 0 and time.time() - t0 > time_budget:
                break
            best = -1
            best_sat = -1
            best_deg = -1
            for v in uncoloured:
                s = len(neighbour_colors[v])
                if s > best_sat or (s == best_sat and degrees[v] > best_deg):
                    best = v
                    best_sat = s
                    best_deg = degrees[v]
            uncoloured.remove(best)
            used = neighbour_colors[best]
            c = 0
            while c in used:
                c += 1
            color[best] = c
            for nb in conflicts[best]:
                if color[nb] == -1:
                    neighbour_colors[nb].add(c)
        # Fill any leftover (shouldn't happen)
        for v in range(m):
            if color[v] == -1:
                used = set()
                for nb in conflicts[v]:
                    if color[nb] != -1:
                        used.add(color[nb])
                c = 0
                while c in used:
                    c += 1
                color[v] = c
        max_c = max(color) if color else 0
        return color, max_c + 1

    def greedy_color(conflicts, m, rng, strategy='largest_first'):
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
        elif strategy == 'largest_random_tie':
            random_vals = [rng.random() for _ in range(m)]
            order = sorted(range(m), key=lambda i: (-len(conflicts[i]), random_vals[i]))
        else:
            order = sorted(range(m), key=lambda i: len(conflicts[i]), reverse=True)
        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if color_of_edge[j] != -1:
                    forbidden.add(color_of_edge[j])
            c = 0
            while c in forbidden:
                c += 1
            color_of_edge[i] = c
        max_c = max(color_of_edge) if color_of_edge else 0
        return color_of_edge, max_c + 1

    def local_search_improve(conflicts, m, colors, max_colors, rng, max_iters=200):
        """Local search with direct reassignment + swap moves to reduce colour count."""
        if m == 0:
            return colors, 0
        cur = colors[:]
        cur_max = max_colors
        t0 = time.time()
        for _ in range(max_iters):
            if time.time() - t0 > 3.0:
                break
            improved = False
            # Pass 1: direct reassignment to lower colour
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
            # Pass 2: try to eliminate highest colour
            if cur_max > 0:
                top = cur_max - 1
                top_verts = [v for v in range(m) if cur[v] == top]
                rng.shuffle(top_verts)
                for v in top_verts:
                    moved = False
                    # direct
                    for c in range(top):
                        ok = True
                        for nb in conflicts[v]:
                            if cur[nb] == c:
                                ok = False
                                break
                        if ok:
                            cur[v] = c
                            moved = True
                            improved = True
                            break
                    if moved:
                        continue
                    # swap: find w with colour c < top such that v can take c
                    # and w can take top (or some colour that v's neighbours allow)
                    for c in range(top):
                        # v wants to go to colour c
                        # find w with cur[w] == c that can go to top
                        candidates = [w for w in range(m) if cur[w] == c]
                        rng.shuffle(candidates)
                        for w in candidates:
                            if w == v:
                                continue
                            # check v -> c is OK given w will leave c
                            ok_v = True
                            for nb in conflicts[v]:
                                if nb == w:
                                    continue
                                if cur[nb] == c:
                                    ok_v = False
                                    break
                            if not ok_v:
                                continue
                            # check w -> top is OK given v will leave top
                            ok_w = True
                            for nb in conflicts[w]:
                                if nb == v:
                                    continue
                                if cur[nb] == top:
                                    ok_w = False
                                    break
                            if not ok_w:
                                continue
                            cur[v] = c
                            cur[w] = top
                            improved = True
                            break
                        if improved:
                            break
            if not improved:
                break
            new_max = max(cur) + 1 if cur else 0
            if new_max < cur_max:
                cur_max = new_max
        return cur, (max(cur) + 1) if cur else 0

    def polish(conflicts, m, colors, rng):
        """Final polish: repeatedly try to lower any edge's colour."""
        cur = colors[:]
        t0 = time.time()
        for _ in range(30):
            if time.time() - t0 > 2.0:
                break
            changed = False
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
                        changed = True
                        break
            if not changed:
                break
        return cur, (max(cur) + 1) if cur else 0

    def color_one_instance(edges, adj, conflicts, m, rng, time_budget=5.0):
        """Try several strategies, return best (colors, score)."""
        if m == 0:
            return [], 0
        best_colors = None
        best_score = float('inf')
        t0 = time.time()
        # DSATUR
        if time.time() - t0 < time_budget:
            dsat_c, dsat_s = dsatur_coloring(conflicts, m, time_budget=1.5)
            if dsat_s < best_score:
                best_score = dsat_s
                best_colors = dsat_c[:]
        # Greedy variants
        for strat in ['largest_first', 'random', 'smallest_first', 'largest_random_tie']:
            if time.time() - t0 > time_budget - 2.0:
                break
            g_c, g_s = greedy_color(conflicts, m, rng, strat)
            if g_s < best_score:
                best_score = g_s
                best_colors = g_c[:]
        # Local search on the best
        if best_colors is not None and time.time() - t0 < time_budget - 1.0:
            ls_c, ls_s = local_search_improve(conflicts, m, best_colors, best_score, rng, max_iters=150)
            if ls_s < best_score:
                best_score = ls_s
                best_colors = ls_c
            # Polish
            pol_c, pol_s = polish(conflicts, m, best_colors, rng)
            if pol_s < best_score:
                best_score = pol_s
                best_colors = pol_c
        return best_colors, best_score

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

    # Pre-build graphs and conflicts
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = make_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = build_conflicts_np(edges, adj, n)
        graphs.append((edges, adj, conflicts, len(edges)))

    best_results = [None] * len(graphs)
    best_scores = [float('inf')] * len(graphs)

    # Initial pass: deterministic DSATUR to guarantee a valid result
    for idx, (edges, adj, conflicts, m) in enumerate(graphs):
        c, s = dsatur_coloring(conflicts, m, time_budget=5.0)
        best_results[idx] = c
        best_scores[idx] = s

    # Iterative improvement with varied random seeds
    iter_num = 0
    while time.time() < deadline - 5.0:
        iter_num += 1
        iter_seed = 50000 + iter_num
        rng = random.Random(iter_seed)
        for idx, (edges, adj, conflicts, m) in enumerate(graphs):
            if time.time() > deadline - 3.0:
                break
            # Per-instance time budget: more time for larger instances
            budget = min(4.0, max(1.5, (deadline - 5.0 - time.time()) / 3.0))
            c, s = color_one_instance(edges, adj, conflicts, m, rng, time_budget=budget)
            if c is not None and s < best_scores[idx]:
                best_scores[idx] = s
                best_results[idx] = c

    # Final polish on each best result
    for idx, (edges, adj, conflicts, m) in enumerate(graphs):
        if time.time() > deadline - 2.0:
            break
        rng = random.Random(99999 + idx)
        pol_c, pol_s = polish(conflicts, m, best_results[idx], rng)
        if pol_s < best_scores[idx]:
            best_scores[idx] = pol_s
            best_results[idx] = pol_c

    # Safety: ensure all results are valid
    for i in range(len(graphs)):
        if best_results[i] is None:
            edges, adj, conflicts, m = graphs[i]
            c, _ = dsatur_coloring(conflicts, m, time_budget=5.0)
            best_results[i] = c

    return best_results
