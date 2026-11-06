import time
import sys
from random import Random
import math

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

        # Phase 1: Greedy coloring to get initial upper bound
        color = [-1] * m
        active = set(range(m))
        current_color_id = 0
        rng = Random(seed * 12345 + 678)

        while active:
            if time.time() > deadline - 10.0:
                break

            # Find a large independent set in the active subgraph
            best_set = set()
            for _ in range(8):
                cand_set = set()
                remaining = set(active)
                while remaining:
                    # Pick vertex with min degree in remaining
                    min_v = -1
                    min_deg = float('inf')
                    vertices = list(remaining)
                    rng.shuffle(vertices)
                    for v in vertices:
                        d_v = sum(1 for u in neigh[v] if u in remaining)
                        if d_v < min_deg:
                            min_deg = d_v
                            min_v = v
                    cand_set.add(min_v)
                    remaining.remove(min_v)
                    for u in neigh[min_v]:
                        if u in remaining:
                            remaining.remove(u)
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

        def count_conflicts(col):
            cnt = 0
            for i in range(m):
                for j in neigh[i]:
                    if j > i and col[i] == col[j]:
                        cnt += 1
            return cnt

        def local_search_reduce(col, K, time_limit):
            if time.time() > time_limit:
                return col, False

            for i in range(m):
                if col[i] >= K:
                    col[i] = rng.randrange(K)

            conflicts = count_conflicts(col)
            if conflicts == 0:
                return col, True

            best_col_ls = col[:]
            best_conflicts = conflicts

            # Precompute neighbor lists for speed
            neigh_local = neigh

            T = 2.0
            cooling = 0.998
            max_iters = 400000
            no_improve = 0
            patience = 5000

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return col, True

                v = rng.randrange(m)
                old_c = col[v]
                new_c = rng.randrange(K)
                if new_c == old_c:
                    continue

                delta = 0
                for u in neigh_local[v]:
                    if col[u] == old_c:
                        delta -= 1
                    if col[u] == new_c:
                        delta += 1

                col[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_col_ls = col[:]
                    no_improve = 0
                elif delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                    no_improve += 1
                else:
                    col[v] = old_c
                    conflicts -= delta
                    no_improve += 1

                T *= cooling

                if no_improve > patience and best_conflicts > 0:
                    # Reset with a more aggressive perturbation
                    for i in range(m):
                        if rng.random() < 0.3:
                            col[i] = rng.randrange(K)
                    conflicts = count_conflicts(col)
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_col_ls = col[:]
                    no_improve = 0
                    T = max(T, 1.5)

            if best_conflicts == 0:
                return best_col_ls, True

            # Final polish pass: greedy improvement with exact delta computation
            if time.time() < time_limit - 1.0:
                improved = True
                while improved and time.time() < time_limit - 0.5:
                    improved = False
                    for i in range(m):
                        if time.time() > time_limit - 0.5:
                            break
                        best_c = col[i]
                        best_delta = 0
                        for c in range(K):
                            if c == col[i]:
                                continue
                            d = 0
                            for u in neigh_local[i]:
                                if col[u] == col[i]:
                                    d -= 1
                                if col[u] == c:
                                    d += 1
                            if d < best_delta:
                                best_delta = d
                                best_c = c
                        if best_delta < 0:
                            col[i] = best_c
                            conflicts += best_delta
                            improved = True
                            if conflicts < best_conflicts:
                                best_conflicts = conflicts
                                best_col_ls = col[:]

                if best_conflicts == 0:
                    return best_col_ls, True

            return col, False

        # Try to reduce K
        for target_K in range(current_K - 1, 2, -1):
            if time.time() > deadline - 3.0:
                break

            success = False
            num_attempts = max(5, int(10 - (current_K - target_K) * 2))
            for attempt in range(num_attempts):
                if time.time() > deadline - 3.0:
                    break

                if attempt == 0:
                    # First attempt: keep structure, just remap colors >= target_K
                    trial = best_color[:]
                    for i in range(m):
                        if trial[i] >= target_K:
                            trial[i] = rng.randrange(target_K)
                elif attempt == 1:
                    # Second attempt: moderate perturbation
                    trial = best_color[:]
                    for i in range(m):
                        if rng.random() < 0.25:
                            trial[i] = rng.randrange(target_K)
                    for i in range(m):
                        if trial[i] >= target_K:
                            trial[i] = rng.randrange(target_K)
                else:
                    # Heavier perturbation for later attempts
                    trial = best_color[:]
                    for i in range(m):
                        if rng.random() < 0.5:
                            trial[i] = rng.randrange(target_K)
                    for i in range(m):
                        if trial[i] >= target_K:
                            trial[i] = rng.randrange(target_K)

                result, ok = local_search_reduce(trial, target_K, deadline - 3.0)
                if ok:
                    best_color = result
                    current_K = target_K
                    success = True
                    break

            if not success:
                # One more try with fresh randomization before giving up on this K
                if time.time() < deadline - 3.0:
                    trial = [rng.randrange(target_K) for _ in range(m)]
                    result, ok = local_search_reduce(trial, target_K, deadline - 3.0)
                    if ok:
                        best_color = result
                        current_K = target_K
                        success = True

            if not success:
                break

        # Final polish: try to merge color classes in the current coloring
        if time.time() < deadline - 2.0:
            final_K = max(best_color) + 1 if best_color else 0
            for merge_target in range(final_K - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                # Try to merge color (merge_target+1) into color merge_target
                merged = best_color[:]
                can_merge = True
                for i in range(m):
                    if merged[i] == merge_target + 1:
                        for j in neigh[i]:
                            if merged[j] == merge_target:
                                can_merge = False
                                break
                    if not can_merge:
                        break
                if can_merge:
                    for i in range(m):
                        if merged[i] == merge_target + 1:
                            merged[i] = merge_target
                    best_color = merged

        return compress(best_color)

    try:
        start = time.time()
        deadline = start + 147.0
        out = []
        # Give more time to harder instances (higher degree)
        time_weights = [1.0, 1.3, 1.6]
        total_weight = sum(time_weights)
        total_time = deadline - start - 2.0  # 2s safety margin
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time() - 1.0
            if remaining < 1.0:
                out.append(fallback()[idx])
                continue
            share = total_time * time_weights[idx] / total_weight
            share = min(share, remaining)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
