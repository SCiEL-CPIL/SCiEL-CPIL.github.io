import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

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

    def build_conflict_masks(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        conflicts = [0] * m
        for i in range(m):
            u, v = edges[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                x, y = edges[j]
                if x == u or x == v or y == u or y == v:
                    mask |= (1 << j)
                elif x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v]:
                    mask |= (1 << j)
            conflicts[i] = mask
        return conflicts

    def greedy_color_bound(candidates_mask, conflict_masks):
        if candidates_mask == 0:
            return 0
        colors = {}
        max_color = 0
        mask = candidates_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            neigh = candidates_mask & ~(conflict_masks[v] | (1 << v))
            forbidden = set()
            for nb, c in colors.items():
                if nb & neigh:
                    forbidden.add(c)
            c = 0
            while c in forbidden:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
            mask ^= lsb
        return max_color + 1

    def max_independent_set(conflict_masks, initial_mask, time_limit):
        best_size = 0
        best_set = 0
        
        def dfs(candidates, current_set, size):
            nonlocal best_size, best_set
            if time.time() > time_limit:
                return
            if candidates == 0:
                if size > best_size:
                    best_size = size
                    best_set = current_set
                return
            bound = greedy_color_bound(candidates, conflict_masks)
            if size + bound <= best_size:
                return
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            new_candidates = candidates & ~(conflict_masks[v] | (1 << v))
            dfs(new_candidates, current_set | lsb, size + 1)
            dfs(candidates & ~lsb, current_set, size)
            
        dfs(initial_mask, 0, 0)
        return best_size, best_set

    def solve_instance(edges, adj, time_budget):
        m = len(edges)
        if m == 0:
            return [], 0
        conflict_masks = build_conflict_masks(edges, adj)
        all_mask = (1 << m) - 1
        remaining_mask = all_mask
        colors = [-1] * m
        color_idx = 0
        
        while remaining_mask:
            time_left = time_budget - (time.time() - start_time)
            if time_left < 0.1:
                break
            step_limit = time.time() + 0.005
            size, set_mask = max_independent_set(conflict_masks, remaining_mask, step_limit)
            if set_mask == 0:
                lsb = remaining_mask & -remaining_mask
                v = lsb.bit_length() - 1
                set_mask = lsb
                size = 1
            temp_mask = set_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                colors[v] = color_idx
                temp_mask ^= lsb
            color_idx += 1
            remove_mask = 0
            temp_mask = set_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                remove_mask |= conflict_masks[v]
                remove_mask |= lsb
                temp_mask ^= lsb
            remaining_mask &= ~remove_mask
        return colors, color_idx

    def greedy_fallback(edges, adj):
        m = len(edges)
        if m == 0:
            return [], 0
        conflict_masks = build_conflict_masks(edges, adj)
        colors = [-1] * m
        remaining = (1 << m) - 1
        color_idx = 0
        while remaining:
            # Pick vertex with min degree in remaining?
            # Just pick first
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            colors[v] = color_idx
            remove = conflict_masks[v] | lsb
            remaining &= ~remove
            color_idx += 1
        return colors, color_idx

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        graphs.append((edges, adj))

    best_results = []
    best_scores = [float('inf')] * len(graphs)

    for idx, (edges, adj) in enumerate(graphs):
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
        try:
            colors, score = solve_instance(edges, adj, time_left)
            if score < best_scores[idx]:
                best_scores[idx] = score
                best_results.append(colors)
        except Exception:
            colors, score = greedy_fallback(edges, adj)
            if score < best_scores[idx]:
                best_scores[idx] = score
                best_results.append(colors)

    for i in range(len(graphs)):
        if i >= len(best_results):
            colors, _ = greedy_fallback(graphs[i][0], graphs[i][1])
            best_results.append(colors)

    return best_results
