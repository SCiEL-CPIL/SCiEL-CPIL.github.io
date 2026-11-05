import time
import random
import sys
from collections import defaultdict

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
        # Find all positions of min_val
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
        from collections import deque
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

    # Generate cycle pool from multiple sources
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
    for c in _find_all_cycles_bfs(adj, n, edge_set, max_cycles=1500, time_limit=2.5):
        _add_to_pool(c)

    # DFS cycles
    for c in _find_cycles_dfs(adj, n, edge_set, max_cycles=2000, time_limit=4.0, rng=rng):
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

    # Objective: minimize pieces = m - sum(|c_i|) + num_cycles
    # = m - sum(|c_i| - 1)
    # Maximize sum(|c_i| - 1) over edge-disjoint cycles

    # Greedy set packing with local search
    def _greedy_and_local(max_restarts=20):
        best_val = -1
        best_selected = []
        best_used = set()

        for restart in range(max_restarts):
            if time.time() > deadline - 0.3:
                break

            rng_r = random.Random(restart * 7919 + 13)
            indices = list(range(len(cycles_pool)))
            rng_r.shuffle(indices)

            # Sort by length descending with some randomness
            order = sorted(indices, key=lambda i: (pool_lengths[i], rng_r.random()), reverse=True)

            remaining = set(edge_set)
            selected = []
            used_edges = set()

            for idx in order:
                if time.time() > deadline - 0.3:
                    break
                if pool_edge_sets[idx].issubset(remaining):
                    selected.append(idx)
                    used_edges |= pool_edge_sets[idx]
                    remaining -= pool_edge_sets[idx]

            # Local search: try swaps
            def _objective(sel):
                return sum(pool_lengths[i] - 1 for i in sel)

            for _ in range(50):
                if time.time() > deadline - 0.3:
                    break
                improved = False
                cur_obj = _objective(selected)
                sel_set = set(selected)

                # Try adding a cycle
                for idx in indices:
                    if idx in sel_set:
                        continue
                    if not pool_edge_sets[idx].issubset(remaining):
                        continue
                    new_sel = selected + [idx]
                    new_obj = cur_obj + pool_lengths[idx] - 1
                    if new_obj > cur_obj:
                        selected = new_sel
                        used_edges |= pool_edge_sets[idx]
                        remaining -= pool_edge_sets[idx]
                        sel_set.add(idx)
                        cur_obj = new_obj
                        improved = True
                        break

                if improved:
                    continue

                # Try swap: remove one, add one
                for j in list(selected):
                    for idx in indices:
                        if idx in sel_set:
                            continue
                        # Remove j, add idx
                        temp_used = used_edges - pool_edge_sets[j]
                        if pool_edge_sets[idx].issubset(temp_used | remaining):
                            # Check if idx's edges are available after removing j
                            avail = (edge_set - used_edges) | pool_edge_sets[j]
                            if pool_edge_sets[idx].issubset(avail):
                                new_sel = [x for x in selected if x != j] + [idx]
                                new_obj = cur_obj - (pool_lengths[j] - 1) + (pool_lengths[idx] - 1)
                                if new_obj > cur_obj:
                                    selected = new_sel
                                    used_edges = (used_edges - pool_edge_sets[j]) | pool_edge_sets[idx]
                                    remaining = edge_set - used_edges
                                    sel_set.discard(j)
                                    sel_set.add(idx)
                                    cur_obj = new_obj
                                    improved = True
                                    break
                    if improved:
                        break

                if not improved:
                    break

            val = _objective(selected)
            if val > best_val:
                best_val = val
                best_selected = list(selected)
                best_used = set(used_edges)

        return best_selected, best_used

    selected, used_edges = _greedy_and_local(max_restarts=25)

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
