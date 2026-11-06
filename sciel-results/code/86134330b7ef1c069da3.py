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

        neigh = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].add(j)
                    neigh[j].add(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].add(j)
                        neigh[j].add(i)

        # Convert to list for faster iteration
        neigh_list = [list(s) for s in neigh]
        deg = [len(neigh_list[i]) for i in range(m)]
        max_deg = max(deg) if deg else 0

        # DSATUR coloring for initial solution
        def dsatur_coloring():
            color = [-1] * m
            color_sets = [set() for _ in range(m)]
            uncolored = list(range(m))
            
            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    sat = len(color_sets[v])
                    dv = len(neigh_list[v])
                    if sat > best_sat or (sat == best_sat and dv > best_deg):
                        best_sat = sat
                        best_deg = dv
                        best_v = v
                
                c = 0
                while c in color_sets[best_v]:
                    c += 1
                color[best_v] = c
                
                for u in neigh_list[best_v]:
                    if color[u] == -1:
                        color_sets[u].add(c)
                
                uncolored.remove(best_v)
            
            return color

        color0 = dsatur_coloring()
        K0 = max(color0) + 1

        rng = Random(seed * 7919 + 42)

        def count_conflicts(color):
            cnt = 0
            for i in range(m):
                for j in neigh_list[i]:
                    if j > i and color[i] == color[j]:
                        cnt += 1
            return cnt

        def local_search_reduce(color, K, time_limit):
            if time.time() > time_limit:
                return color, False

            for i in range(m):
                if color[i] >= K:
                    color[i] = rng.randrange(K)

            conflicts = count_conflicts(color)
            if conflicts == 0:
                return color, True

            best_color_ls = color[:]
            best_conflicts = conflicts

            T = 1.0
            T_min = 0.005
            cooling = 0.998
            max_iters = 300000

            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                if conflicts == 0:
                    return color, True

                v = rng.randrange(m)
                old_c = color[v]
                new_c = rng.randrange(K)
                if new_c == old_c:
                    continue

                delta = 0
                for u in neigh_list[v]:
                    if color[u] == old_c:
                        delta -= 1
                    if color[u] == new_c:
                        delta += 1

                color[v] = new_c
                conflicts += delta

                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_color_ls = color[:]
                elif delta <= 0 or rng.random() < math.exp(-delta / max(T, 0.001)):
                    pass
                else:
                    color[v] = old_c
                    conflicts -= delta

                T *= cooling

            if best_conflicts == 0:
                return best_color_ls, True
            return color, False

        def mis_peeling(remaining, time_limit):
            """Extract a maximum independent set from remaining vertices using greedy + local search."""
            if not remaining:
                return []
            
            remaining_set = set(remaining)
            # Greedy: pick vertex with minimum degree in induced subgraph
            selected = []
            unselected = set(remaining)
            
            while unselected and time.time() < time_limit:
                # Find vertex with minimum degree in remaining subgraph
                min_deg = float('inf')
                best_v = None
                for v in unselected:
                    d_in = len([u for u in unselected if u in neigh_list[v]])
                    if d_in < min_deg:
                        min_deg = d_in
                        best_v = v
                
                if best_v is None:
                    break
                    
                selected.append(best_v)
                unselected.discard(best_v)
                # Remove neighbors of best_v from unselected
                to_remove = []
                for u in unselected:
                    if u in neigh_list[best_v]:
                        to_remove.append(u)
                for u in to_remove:
                    unselected.discard(u)
            
            return selected

        def try_reduce_with_mis(current_color, current_K, time_limit):
            """Try to reduce colour count by extracting MIS for each colour class."""
            if time.time() > time_limit:
                return None, False
            
            # Group vertices by colour
            color_groups = {}
            for i in range(m):
                c = current_color[i]
                if c not in color_groups:
                    color_groups[c] = []
                color_groups[c].append(i)
            
            # Try to merge colour classes using MIS
            new_color = current_color[:]
            new_K = current_K
            
            # Sort colours by size (largest first)
            sorted_colors = sorted(color_groups.keys(), key=lambda c: -len(color_groups[c]))
            
            # For each colour, try to find a larger independent set
            for c in sorted_colors:
                if time.time() > time_limit - 2:
                    break
                group = color_groups[c]
                if len(group) < 2:
                    continue
                
                # Try to extend this independent set
                # Find vertices not in this group that can be added
                candidates = [v for v in range(m) if v not in set(group) and 
                             all(u not in neigh_list[v] for u in group)]
                
                if candidates:
                    # Try to add some candidates
                    added = 0
                    for v in candidates:
                        if time.time() > time_limit - 2:
                            break
                        # Check if v can be added (no conflicts with current group)
                        can_add = all(u not in neigh_list[v] for u in group)
                        if can_add:
                            new_color[v] = c
                            group.append(v)
                            added += 1
                            # Remove v from other colour groups
                            # (handled implicitly since we're building new_color)
            
            # Compress and check
            used_colors = set(new_color)
            if len(used_colors) < new_K:
                # Compress
                mapping = {}
                compressed = [0] * m
                nxt = 0
                for i in range(m):
                    c = new_color[i]
                    if c not in mapping:
                        mapping[c] = nxt
                        nxt += 1
                    compressed[i] = mapping[c]
                if count_conflicts(compressed) == 0:
                    return compressed, True
            
            return None, False

        current_color = color0[:]
        current_K = K0

        for target_K in range(current_K - 1, 2, -1):
            if time.time() > deadline - 5.0:
                break

            success = False
            # Try MIS-based reduction first
            result, ok = try_reduce_with_mis(current_color, current_K, deadline - 5.0)
            if ok and max(result) + 1 <= target_K:
                current_color = result
                current_K = max(result) + 1
                success = True
            
            if not success:
                for attempt in range(8):
                    if time.time() > deadline - 5.0:
                        break
                    trial = current_color[:]
                    for i in range(m):
                        if rng.random() < 0.3:
                            trial[i] = rng.randrange(target_K)
                    for i in range(m):
                        if trial[i] >= target_K:
                            trial[i] = rng.randrange(target_K)

                    result, ok = local_search_reduce(trial, target_K, deadline - 5.0)
                    if ok:
                        current_color = result
                        current_K = target_K
                        success = True
                        break

            if not success:
                for attempt in range(4):
                    if time.time() > deadline - 5.0:
                        break
                    trial = [rng.randrange(target_K) for _ in range(m)]
                    result, ok = local_search_reduce(trial, target_K, deadline - 5.0)
                    if ok:
                        current_color = result
                        current_K = target_K
                        success = True
                        break

            if not success:
                break

        return compress(current_color)

    try:
        start = time.time()
        deadline = start + 140.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = remaining / (len(specs) - idx)
            share = min(share, 45.0)
            share = max(share, 1.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
