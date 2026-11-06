"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _remove_edge(adj, remaining, u, v):
    e = (min(u, v), max(u, v))
    remaining.discard(e)
    if v in adj[u]:
        adj[u].remove(v)
    if u in adj[v]:
        adj[v].remove(u)

def _find_cycle_in_adj(adj, n, seed, deadline=None):
    """
    Find a cycle in the adjacency structure.
    Uses multiple strategies:
    1. DFS-based cycle detection
    2. Greedy longest path extension
    Returns a list of vertices forming a cycle, or empty list if none found.
    """
    if not adj:
        return []

    vertices = [v for v in adj if adj[v]]
    if len(vertices) < 3:
        return []

    best_cycle = []
    rng = random.Random(seed)

    # Strategy 1: DFS-based cycle detection
    # Try to find cycles by doing DFS from various starting points
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:8] + vertices[:8]))
    rng.shuffle(starts)

    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # DFS to find a cycle
        visited = {start}
        path = [start]
        parent = {start: -1}
        found_cycle = False

        def dfs(u):
            nonlocal found_cycle
            if found_cycle:
                return
            for v in adj[u]:
                if v == start and len(path) >= 3:
                    # Found a cycle back to start
                    found_cycle = True
                    return
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    path.append(v)
                    dfs(v)
                    if not found_cycle:
                        path.pop()
                    else:
                        return

        dfs(start)

        if found_cycle:
            # Extract the cycle from the path
            # The cycle is from start to the last vertex that connects back to start
            cycle_vertices = []
            for v in path:
                cycle_vertices.append(v)
                if v == start:
                    break
            # The cycle should be: start -> ... -> last -> start
            # But we need to find where the cycle actually is
            # Let's reconstruct: find the index where we first revisit a node in path
            # Actually, let's use a different approach: track the stack properly

        # Strategy 2: Greedy longest path extension
        path = [start]
        visited = {start}

        while True:
            u = path[-1]
            candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
            if not candidates:
                break
            candidates.sort(reverse=True)
            top_k = min(3, len(candidates))
            _, next_v = rng.choice(candidates[:top_k])
            path.append(next_v)
            visited.add(next_v)

        # Check if we can close the cycle directly
        if len(path) >= 3 and start in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)

        # Look for any back-edge (path[i], path[j]) with i < j-1 to form a cycle
        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(len(path) - 1, i + 1, -1):
                if path[i] in adj[path[j]]:
                    cycle = path[i:j + 1]
                    if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break

    # Strategy 3: Try to find triangles quickly
    for v in vertices:
        if deadline is not None and time.time() > deadline:
            break
        if len(adj[v]) < 2:
            continue
        adj_set = set(adj[v])
        for u in adj[v]:
            if u <= v:
                continue
            for w in adj[u]:
                if w > u and w in adj_set:
                    triangle = [v, u, w]
                    if len(triangle) > len(best_cycle):
                        best_cycle = triangle

    return best_cycle

def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []

    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    global_deadline = start_time + time_limit

    def greedy_decompose(remaining_edges, n, seed_offset=0, strategy=0):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(list)
        for u, v in remaining:
            adj[u].append(v)
            adj[v].append(u)

        cycle_counter = 0
        while remaining:
            if time.time() > global_deadline:
                break

            # Choose cycle finding strategy
            if strategy == 0:
                cycle = _find_cycle_in_adj(adj, n, seed + seed_offset * 1000 + cycle_counter, deadline=global_deadline)
            else:
                # Alternate strategies for diversity
                if cycle_counter % 2 == 0:
                    cycle = _find_cycle_in_adj(adj, n, seed + seed_offset * 1000 + cycle_counter, deadline=global_deadline)
                else:
                    # Try a different random seed
                    cycle = _find_cycle_in_adj(adj, n, seed + seed_offset * 1000 + cycle_counter + 500, deadline=global_deadline)

            if cycle and len(cycle) >= 3:
                # Verify all cycle edges exist in remaining
                cycle_edges = set()
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e not in remaining:
                        valid = False
                        break
                    cycle_edges.add(e)

                if valid:
                    for e in cycle_edges:
                        u, v = e
                        _remove_edge(adj, remaining, u, v)
                    pieces.append(cycle)
                    cycle_counter += 1
                else:
                    # Cycle invalid, fall through to single edge
                    e = next(iter(remaining))
                    u, v = e
                    _remove_edge(adj, remaining, u, v)
                    pieces.append([u, v])
            else:
                # No cycle found, take a single edge
                # Prefer edges where both endpoints have low degree
                best_edge = None
                best_priority = -1
                for u, v in remaining:
                    deg_u = len(adj[u])
                    deg_v = len(adj[v])
                    priority = -(deg_u + deg_v)
                    if priority > best_priority:
                        best_priority = priority
                        best_edge = (u, v)

                if best_edge is None:
                    best_edge = next(iter(remaining))

                u, v = best_edge
                _remove_edge(adj, remaining, u, v)
                pieces.append([u, v])

        return pieces

    # Multi-start search with diverse strategies
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > global_deadline:
            break

        strategy = restart % 2
        pieces = greedy_decompose(edges, n, restart, strategy)
        current_count = len(pieces)

        if current_count < best_count:
            best_count = current_count
            best_pieces = pieces

    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_time_limit = 145.0
    start = time.time()
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        elapsed = time.time() - start
        remaining_time = max(1.0, total_time_limit - elapsed - 1.0)
        pieces = _decompose(edges, n, seed, time_limit=remaining_time)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
