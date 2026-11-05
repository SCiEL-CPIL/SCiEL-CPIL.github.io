import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_all_cycles_bfs(adj, n, edge_set, max_cycles=2000, time_limit=3.0):
    """
    Find many simple cycles using BFS from each edge.
    For each edge (u,v), find shortest path from u to v not using (u,v).
    Each such path + (u,v) forms a cycle.
    """
    start_time = time.time()
    cycles = []
    seen_canonical = set()

    def _canonical(cycle):
        if len(cycle) < 3:
            return None
        min_val = min(cycle)
        idxs = [i for i, v in enumerate(cycle) if v == min_val]
        candidates = []
        for idx in idxs:
            c1 = cycle[idx:] + cycle[:idx]
            c2 = list(reversed(c1))
            candidates.append(tuple(c1) if c1 < c2 else tuple(c2))
        return min(candidates)

    def _add_cycle(cycle):
        if len(cycle) < 3:
            return
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((a, b)))
            if e not in edge_set:
                return
        key = _canonical(cycle)
        if key and key not in seen_canonical:
            seen_canonical.add(key)
            cycles.append(list(cycle))

    # For each edge, BFS to find path between its endpoints
    for u, v in edge_set:
        if time.time() - start_time > time_limit:
            break
        # BFS from u to v, not using edge (u,v)
        parent = {u: None}
        queue = deque([u])
        found = False
        while queue and not found:
            if time.time() - start_time > time_limit:
                break
            cur = queue.popleft()
            for nb in adj[cur]:
                if nb == v and cur == u:
                    continue  # skip the direct edge
                if nb not in parent:
                    parent[nb] = cur
                    if nb == v:
                        found = True
                        break
                    queue.append(nb)
        if found:
            # Reconstruct path
            path = [v]
            cur = v
            while cur != u:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            _add_cycle(path)

    return cycles


def _find_cycles_dfs(adj, n, edge_set, max_cycles=3000, time_limit=5.0, rng=None):
    """
    Find cycles using randomized DFS with backtracking.
    """
    if rng is None:
        rng = random.Random(42)
    start_time = time.time()
    cycles = []
    seen_canonical = set()

    def _canonical(cycle):
        if len(cycle) < 3:
            return None
        min_val = min(cycle)
        idxs = [i for i, v in enumerate(cycle) if v == min_val]
        candidates = []
        for idx in idxs:
            c1 = cycle[idx:] + cycle[:idx]
            c2 = list(reversed(c1))
            candidates.append(tuple(c1) if c1 < c2 else tuple(c2))
        return min(candidates)

    def _add_cycle(cycle):
        if len(cycle) < 3:
            return
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((a, b)))
            if e not in edge_set:
                return
        key = _canonical(cycle)
        if key and key not in seen_canonical:
            seen_canonical.add(key)
            cycles.append(list(cycle))

    # DFS from each vertex
    for start in range(n):
        if time.time() - start_time > time_limit:
            break
        if not adj[start]:
            continue
        for attempt in range(5):
            visited = {start}
            path = [start]
            stack = [(start, iter(sorted(adj[start], key=lambda x: len(adj[x]))))]
            found_any = False
            while stack:
                if time.time() - start_time > time_limit:
                    break
                node, it = stack[-1]
                advanced = False
                for nb in it:
                    if nb == start and len(path) >= 3:
                        _add_cycle(path[:])
                        found_any = True
                        break
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        stack.append((nb, iter(sorted(adj[nb], key=lambda x: len(adj[x])))))
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
                    if path:
                        path.pop()
                        visited.discard(node)
                    if not path:
                        break
                if found_any and len(cycles) >= max_cycles:
                    break

    return cycles


def _find_cycles_random_walk(adj, n, edge_set, max_cycles=1000, time_limit=3.0, rng=None):
    """
    Find cycles by doing random walks and trying to close the loop.
    """
    if rng is None:
        rng = random.Random(999)
    start_time = time.time()
    cycles = []
    seen_canonical = set()

    def _canonical(cycle):
        if len(cycle) < 3:
            return None
        min_val = min(cycle)
        idxs = [i for i, v in enumerate(cycle) if v == min_val]
        candidates = []
        for idx in idxs:
            c1 = cycle[idx:] + cycle[:idx]
            c2 = list(reversed(c1))
            candidates.append(tuple(c1) if c1 < c2 else tuple(c2))
        return min(candidates)

    def _add_cycle(cycle):
        if len(cycle) < 3:
            return
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((a, b)))
            if e not in edge_set:
                return
        key = _canonical(cycle)
        if key and key not in seen_canonical:
            seen_canonical.add(key)
            cycles.append(list(cycle))

    for _ in range(max_cycles):
        if time.time() - start_time > time_limit:
            break
        start = rng.randrange(n)
        if not adj[start]:
            continue
        path = [start]
        visited = {start}
        success = False
        for _ in range(20): # Max length 20
            cur = path[-1]
            neighbors = list(adj[cur])
            rng.shuffle(neighbors)
            moved = False
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    _add_cycle(path)
                    success = True
                    break
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    moved = True
                    break
            if success or not moved:
                break
    return cycles


def _find_cycles_triangles(adj, n, edge_set):
    """Find all triangles."""
    cycles = []
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    cycles.append([u, v, w])
    return cycles


def _solve_one(edges, n, deadline):
    if not edges:
        return []

    m = len(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edge_set, n)

    if m < 3:
        return [list(e) for e in edges]

    # Generate cycle pool
    rng = random.Random(12345)
    cycles_pool = []
    seen = set()

    def _add_to_pool(cycle):
        if len(cycle) < 3:
            return
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((a, b)))
            if e not in edge_set:
                return
        min_val = min(cycle)
        idxs = [i for i, v in enumerate(cycle) if v == min_val]
        candidates = []
        for idx in idxs:
            c1 = cycle[idx:] + cycle[:idx]
            c2 = list(reversed(c1))
            candidates.append(tuple(c1) if c1 < c2 else tuple(c2))
        key = min(candidates)
        if key not in seen:
            seen.add(key)
            cycles_pool.append(list(cycle))

    # Triangles
    for c in _find_cycles_triangles(adj, n, edge_set):
        _add_to_pool(c)

    # BFS cycles
    for c in _find_all_cycles_bfs(adj, n, edge_set, max_cycles=1500, time_limit=2.0):
        _add_to_pool(c)

    # DFS cycles
    for c in _find_cycles_dfs(adj, n, edge_set, max_cycles=2000, time_limit=3.0, rng=rng):
        _add_to_pool(c)
        
    # Random walk cycles
    for c in _find_cycles_random_walk(adj, n, edge_set, max_cycles=1000, time_limit=2.0, rng=rng):
        _add_to_pool(c)

    if not cycles_pool:
        return [list(e) for e in edges]

    # Precompute
    pool_edge_sets = []
    pool_lengths = []
    for c in cycles_pool:
        es = set()
        L = len(c)
        for i in range(L):
            a = c[i]
            b = c[(i + 1) % L]
            es.add(tuple(sorted((a, b))))
        pool_edge_sets.append(es)
        pool_lengths.append(L)

    # Simulated Annealing
    def _solve_sa():
        if time.time() > deadline - 1.0:
            return None

        # Initial solution: Greedy by length descending
        indices = list(range(len(cycles_pool)))
        # Sort by length desc
        indices_sorted = sorted(indices, key=lambda i: pool_lengths[i], reverse=True)
        
        current_selected = []
        current_used = set()
        current_remaining = set(edge_set)
        current_obj = 0

        for idx in indices_sorted:
            if time.time() > deadline - 1.0:
                break
            if pool_edge_sets[idx].issubset(current_remaining):
                current_selected.append(idx)
                current_used |= pool_edge_sets[idx]
                current_remaining -= pool_edge_sets[idx]
                current_obj += pool_lengths[idx] - 1

        best_selected = list(current_selected)
        best_obj = current_obj
        best_used = set(current_used)

        T = 5.0
        T_min = 0.01
        alpha = 0.95
        rng_sa = random.Random(54321)
        
        # Precompute list of indices for random selection
        all_indices = list(range(len(cycles_pool)))
        
        steps = 0
        max_steps = 5000
        
        while T > T_min and steps < max_steps:
            if time.time() > deadline - 0.5:
                break
            
            steps += 1
            move_type = rng_sa.random()
            
            if move_type < 0.4:
                # Add a random cycle
                idx = rng_sa.choice(all_indices)
                if idx not in current_selected:
                    if pool_edge_sets[idx].issubset(current_remaining):
                        new_obj = current_obj + pool_lengths[idx] - 1
                        delta = new_obj - current_obj
                        if delta > 0 or rng_sa.random() < pow(2.718, delta / T):
                            current_selected.append(idx)
                            current_used |= pool_edge_sets[idx]
                            current_remaining -= pool_edge_sets[idx]
                            current_obj = new_obj
                            if current_obj > best_obj:
                                best_obj = current_obj
                                best_selected = list(current_selected)
                                best_used = set(current_used)
            elif move_type < 0.8:
                # Remove a random cycle
                if current_selected:
                    idx = rng_sa.choice(current_selected)
                    new_obj = current_obj - (pool_lengths[idx] - 1)
                    delta = new_obj - current_obj
                    if delta > 0 or rng_sa.random() < pow(2.718, delta / T):
                        current_selected.remove(idx)
                        current_used -= pool_edge_sets[idx]
                        current_remaining |= pool_edge_sets[idx]
                        current_obj = new_obj
            else:
                # Swap: Remove one, add one
                if current_selected:
                    idx_remove = rng_sa.choice(current_selected)
                    idx_add = rng_sa.choice(all_indices)
                    if idx_add not in current_selected:
                        # Check if idx_add fits if we remove idx_remove
                        temp_remaining = current_remaining | pool_edge_sets[idx_remove]
                        if pool_edge_sets[idx_add].issubset(temp_remaining):
                            new_obj = current_obj - (pool_lengths[idx_remove] - 1) + (pool_lengths[idx_add] - 1)
                            delta = new_obj - current_obj
                            if delta > 0 or rng_sa.random() < pow(2.718, delta / T):
                                current_selected.remove(idx_remove)
                                current_used -= pool_edge_sets[idx_remove]
                                current_remaining |= pool_edge_sets[idx_remove]
                                
                                current_selected.append(idx_add)
                                current_used |= pool_edge_sets[idx_add]
                                current_remaining -= pool_edge_sets[idx_add]
                                
                                current_obj = new_obj
                                if current_obj > best_obj:
                                    best_obj = current_obj
                                    best_selected = list(current_selected)
                                    best_used = set(current_used)
            
            T *= alpha

        return best_selected, best_used

    result = _solve_sa()
    if result:
        selected, used_edges = result
    else:
        # Fallback to simple greedy
        selected = []
        used_edges = set()
        remaining = set(edge_set)
        indices = sorted(range(len(cycles_pool)), key=lambda i: pool_lengths[i], reverse=True)
        for idx in indices:
            if pool_edge_sets[idx].issubset(remaining):
                selected.append(idx)
                used_edges |= pool_edge_sets[idx]
                remaining -= pool_edge_sets[idx]

    remaining = edge_set - used_edges
    pieces = [list(cycles_pool[i]) for i in selected]
    for e in remaining:
        pieces.append(list(e))

    return pieces


def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)

        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
