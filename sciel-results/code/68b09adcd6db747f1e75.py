import time
import sys
from random import Random
import math
from collections import Counter

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build conflict graph on edges
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Conflict: two edges conflict if they share a vertex OR are joined by an edge
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)

        deg = [len(neigh[i]) for i in range(m)]
        neigh_sets = [set(neigh[i]) for i in range(m)]

        # Phase 1: Greedy coloring with better independent set extraction
        color = [-1] * m
        active = set(range(m))
        current_color_id = 0
        rng = Random(seed * 12345 + 678)

        while active:
            if time.time() > deadline - 10.0:
                break

            # Find a large independent set using greedy min-degree with multiple restarts
            best_set = set()
            for restart in range(12):
                cand_set = set()
                remaining = set(active)
                # Use a priority queue-like approach: always pick min-degree vertex
                while remaining:
                    # Find vertex with minimum degree in remaining subgraph
                    min_v = None
                    min_deg = float('inf')
                    for v in remaining:
                        d_v = len(neigh_sets[v] & remaining)
                        if d_v < min_deg:
                            min_deg = d_v
                            min_v = v
                    cand_set.add(min_v)
                    remaining.remove(min_v)
                    remaining -= neigh_sets[min_v]
                if len(cand_set) > len(best_set):
                    best_set = cand_set

            for v in best_set:
                color[v] = current_color_id
            active -= best_set
            current_color_id += 1

        if active:
            for v in sorted(active, key=lambda x: -deg[x]):
                used = set()
                for u in neigh[v]:
                    if color[u] != -1:
                        used.add(color[u])
                c = 0
                while c in used:
                    c += 1
                color[v] = c

        current_K = max(color) + 1 if color else 0
        best_color = color[:]

        def count_conflicts_fast(col):
            """Count conflicts using color class sizes and neighbor checks."""
            cnt = 0
            for i in range(m):
                ci = col[i]
                for j in neigh[i]:
                    if j > i and col[j] == ci:
                        cnt += 1
            return cnt

        def local_search_reduce(col, K, time_limit):
            """Simulated annealing to reduce conflicts to zero with K colors."""
            if time.time() > time_limit:
                return col, False

            # Ensure all colors are in range
            for i in range(m):
                if col[i] >= K:
                    col[i] = rng.randrange(K)

            conflicts = count_conflicts_fast(col)
            if conflicts == 0:
                return col, True

            best_col_ls = col[:]
            best_conflicts = conflicts

            T = max(1.0, 2.0 * conflicts / max(1, m))
            cooling = 0.995
            max_iters = 500000
            stall_count = 0
            stall_threshold = 20000

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return col, True

                # Pick a random vertex that has a conflict (biased selection)
                if rng.random() < 0.7:
                    # Pick from vertices involved in conflicts
                    conflicted = []
                    for i in range(m):
                        ci = col[i]
                        for j in neigh[i]:
                            if j > i and col[j] == ci:
                                conflicted.append(i)
                                conflicted.append(j)
                                break
                    if conflicted:
                        v = conflicted[rng.randrange(len(conflicted))]
                    else:
                        v = rng.randrange(m)
                else:
                    v = rng.randrange(m)

                old_c = col[v]
                new_c = rng.randrange(K)
                if new_c == old_c:
                    continue

                delta = 0
                for u in neigh[v]:
                    if col[u] == old_c:
                        delta -= 1
                    if col[u] == new_c:
                        delta += 1

                col[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_col_ls = col[:]
                    stall_count = 0
                else:
                    stall_count += 1

                if delta > 0 and rng.random() >= math.exp(-delta / max(T, 0.001)):
                    col[v] = old_c
                    conflicts -= delta

                T *= cooling

                # Occasionally reset to best solution if stalled
                if stall_count > stall_threshold and best_conflicts < conflicts:
                    col = best_col_ls[:]
                    conflicts = best_conflicts
                    stall_count = 0
                    T = max(T, 0.5)

            if best_conflicts == 0:
                return best_col_ls, True
            return col, False

        # Try to reduce K: try from current_K-1 down to a lower bound
        # Lower bound: max degree in conflict graph + 1
        lb = max(deg) + 1 if m > 0 else 0
        lb = max(lb, 3)  # at least 3 colors for non-trivial graphs

        for target_K in range(current_K - 1, max(lb, 2) - 1, -1):
            if time.time() > deadline - 5.0:
                break

            success = False
            # More attempts for smaller K reductions
            num_attempts = 20 if target_K >= lb + 2 else 30
            for attempt in range(num_attempts):
                if time.time() > deadline - 5.0:
                    break
                trial = best_color[:]
                # Randomize: keep some structure, randomize the rest
                if attempt == 0:
                    # Just clamp to target_K
                    for i in range(m):
                        if trial[i] >= target_K:
                            trial[i] = rng.randrange(target_K)
                else:
                    # More aggressive randomization
                    frac = 0.3 + 0.5 * (attempt / num_attempts)
                    for i in range(m):
                        if rng.random() < frac:
                            trial[i] = rng.randrange(target_K)
                    for i in range(m):
                        if trial[i] >= target_K:
                            trial[i] = rng.randrange(target_K)

                result, ok = local_search_reduce(trial, target_K, deadline - 5.0)
                if ok:
                    best_color = result
                    current_K = target_K
                    success = True
                    break

            if not success:
                # Try one more time with a completely fresh random assignment
                trial = [rng.randrange(target_K) for _ in range(m)]
                result, ok = local_search_reduce(trial, target_K, deadline - 5.0)
                if ok:
                    best_color = result
                    current_K = target_K
                else:
                    break

        return compress(best_color)

    try:
        start = time.time()
        deadline = start + 145.0
        out = []
        # Allocate time: give more time to harder instances (larger n/d)
        # Instance sizes: (40,4), (60,5), (80,6)
        # Approximate difficulty weights based on edge count ~ n*d/2
        weights = [40*4, 60*5, 80*6]
        total_w = sum(weights)
        avail = deadline - start - 5.0  # 5s safety margin
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time() - 5.0
            if remaining <= 0:
                break
            # Proportional allocation with minimum guarantee
            share = (weights[idx] / total_w) * avail
            share = max(share, 5.0)
            share = min(share, remaining)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
