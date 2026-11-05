"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

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

    def build_conflict_sets(edges, adj):
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        n_edges = len(edges)
        conflict_sets = [set() for _ in range(n_edges)]
        for i, e in enumerate(edges):
            u, v = e
            for w in adj[u]:
                ee = (min(u, w), max(u, w))
                if ee in edge_to_idx:
                    j = edge_to_idx[ee]
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
            for w in adj[v]:
                ee = (min(v, w), max(v, w))
                if ee in edge_to_idx:
                    j = edge_to_idx[ee]
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
            for w in adj[u]:
                if w == v:
                    continue
                for x in adj[w]:
                    if x == u:
                        continue
                    ee = (min(w, x), max(w, x))
                    if ee in edge_to_idx:
                        j = edge_to_idx[ee]
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == v:
                        continue
                    ee = (min(w, x), max(w, x))
                    if ee in edge_to_idx:
                        j = edge_to_idx[ee]
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)
        return conflict_sets

    def greedy_color(edges, adj, conflict_sets, ordering_seed=None):
        n_edges = len(edges)
        if n_edges == 0:
            return [], {}, 0
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        deg = [len(cs) for cs in conflict_sets]
        if ordering_seed is not None:
            r = random.Random(ordering_seed)
            order = sorted(range(n_edges), key=lambda i: -(deg[i] + r.random() * 0.01))
        else:
            order = sorted(range(n_edges), key=lambda i: -deg[i])
        color = [None] * n_edges
        color_used = {}
        next_color = 0
        for idx in order:
            taken = set()
            for j in conflict_sets[idx]:
                if color[j] is not None:
                    taken.add(color[j])
            c = next_color
            while c in taken:
                c += 1
            color[idx] = c
            if c >= next_color:
                next_color = c + 1
            color_used.setdefault(c, []).append(idx)
        return color, color_used, next_color

    def try_merge(color, color_used, conflict_sets):
        n_colors = len(color_used)
        if n_colors <= 1:
            return color, color_used, n_colors
        changed = True
        while changed:
            changed = False
            n_colors = len(color_used)
            for c1 in range(n_colors):
                if c1 not in color_used:
                    continue
                for c2 in range(c1 + 1, n_colors):
                    if c2 not in color_used:
                        continue
                    compatible = True
                    for i in color_used[c1]:
                        for j in color_used[c2]:
                            if j in conflict_sets[i]:
                                compatible = False
                                break
                        if not compatible:
                            break
                    if compatible:
                        for i in color_used[c2]:
                            color[i] = c1
                        color_used[c1].extend(color_used[c2])
                        del color_used[c2]
                        changed = True
                        break
                if changed:
                    break
        if color_used:
            new_map = {}
            new_next = 0
            for c in sorted(color_used.keys()):
                new_map[c] = new_next
                new_next += 1
            new_color = [new_map[c] for c in color]
            new_cused = {}
            for c, idxs in color_used.items():
                new_cused[new_map[c]] = idxs
            return new_color, new_cused, new_next
        return color, color_used, 0

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    results = []

    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        conflict_sets = build_conflict_sets(edges, adj)
        n_edges = len(edges)

        best_color = None
        best_nc = float('inf')

        for trial in range(25):
            if time.time() > deadline - 5.0:
                break
            os = seed * 10000 + trial * 137
            color, cused, nc = greedy_color(edges, adj, conflict_sets, os)
            color_m, cused_m, nc_m = try_merge(color, cused, conflict_sets)
            if nc_m < best_nc:
                best_nc = nc_m
                best_color = color_m

        if best_color is None:
            color, cused, nc = greedy_color(edges, adj, conflict_sets, None)
            best_color = color
            best_nc = nc

        results.append([best_color[i] for i in range(n_edges)])

    return results
# EVOLVE-BLOCK-END
