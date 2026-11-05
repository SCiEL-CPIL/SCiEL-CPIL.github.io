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


def _cycle_edges(cycle):
    """Return set of edges for a cycle."""
    edges = set()
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        edges.add(tuple(sorted((a, b))))
    return edges


def _find_longest_cycle(adj, n, edge_set, rng, max_attempts=100, time_limit=1.0):
    """
    Find a long cycle in the graph using random DFS with restarts.
    Returns the longest cycle found (list of vertices) or None.
    """
    start_time = time.time()
    best_cycle = None
    best_len = 0

    for _ in range(max_attempts):
        if time.time() - start_time > time_limit:
            break

        # Pick a random start vertex with degree >= 2
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            break

        start = rng.choice(candidates)
        visited = {start}
        path = [start]
        current = start
        found = False

        for depth in range(n):
            if time.time() - start_time > time_limit:
                break
            neighbors = list(adj[current])
            # Check if we can close a cycle
            if start in neighbors and len(path) >= 3:
                cycle = path[:]
                if len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                found = True
                break

            unvisited = [v for v in neighbors if v not in visited]
            if not unvisited:
                break

            # Try different strategies for choosing next vertex
            if rng.random() < 0.3:
                # Random
                next_v = rng.choice(unvisited)
            else:
                # Prefer high-degree vertices
                scored = sorted(unvisited, key=lambda v: len(adj[v]), reverse=True)
                next_v = scored[0]

            path.append(next_v)
            visited.add(next_v)
            current = next_v

    return best_cycle


def _find_all_cycles(adj, n, edge_set, rng, time_limit=2.0):
    """
    Find multiple cycles in the graph using various strategies.
    Returns a list of cycles (each a list of vertices).
    """
    start_time = time.time()
    cycles = []
    seen_canonical = set()

    def _canonical(cycle):
        if len(cycle) < 3:
            return None
        min_val = min(cycle)
        min_idx = cycle.index(min_val)
        c1 = cycle[min_idx:] + cycle[:min_idx]
        c2 = list(reversed(c1))
        return tuple(c1) if c1 < c2 else tuple(c2)

    def _add_cycle(cycle):
        if len(cycle) < 3:
            return False
        # Verify all edges exist
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((a, b)))
            if e not in edge_set:
                return False
        key = _canonical(cycle)
        if key and key not in seen_canonical:
            seen_canonical.add(key)
            cycles.append(list(cycle))
            return True
        return False

    # Strategy 1: Find triangles (3-cycles)
    for u in range(n):
        if time.time() - start_time > time_limit:
            return cycles
        for v in adj[u]:
            if u >= v:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    _add_cycle([u, v, w])

    # Strategy 2: Random DFS for longer cycles
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            break
        start = rng.choice(candidates)
        visited = {start}
        path = [start]
        current = start
        for depth in range(n):
            if time.time() - start_time > time_limit:
                break
            neighbors = list(adj[current])
            if start in neighbors and len(path) >= 3:
                _add_cycle(path[:])
                break
            unvisited = [v for v in neighbors if v not in visited]
            if not unvisited:
                break
            if rng.random() < 0.5:
                next_v = rng.choice(unvisited)
            else:
                scored = sorted(unvisited, key=lambda v: len(adj[v]), reverse=True)
                next_v = scored[0]
            path.append(next_v)
            visited.add(next_v)
            current = next_v

    # Strategy 3: BFS-based cycle detection
    for start in range(n):
        if time.time() - start_time > time_limit:
            break
        if len(adj[start]) < 2:
            continue
        visited = {start}
        parent = {start: -1}
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v == start and len(parent) >= 3:
                    # Reconstruct cycle
                    cycle = []
                    cur = u
                    while cur != start:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(start)
                    cycle.reverse()
                    _add_cycle(cycle)
                elif v not in visited:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)

    return cycles


def _solve_one(edges, n, deadline):
    """
    Solve one instance: decompose edges into minimum number of pieces.
    Uses iterative cycle extraction with local search improvements.
    """
    if not edges:
        return []

    m = len(edges)
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    if m < 3:
        return [list(e) for e in edges]

    # If it's a forest (no cycles), all edges are single pieces
    if m < n:
        return [list(e) for e in edges]

    best_pieces = None
    best_count = float('inf')

    max_restarts = 60
    for attempt in range(max_restarts):
        if time.time() > deadline - 0.3:
            break

        rng_attempt = random.Random(attempt * 251 + 17)
        remaining = set(edge_set)
        current_adj = [set() for _ in range(n)]
        for u, v in remaining:
            current_adj[u].add(v)
            current_adj[v].add(u)

        selected_cycles = []
        used_edges = set()

        # Phase 1: Greedy cycle extraction
        while time.time() < deadline - 0.5:
            # Find cycles in the current remaining graph
            cycles = _find_all_cycles(current_adj, n, remaining, rng_attempt, time_limit=1.0)

            if not cycles:
                break

            # Sort by length descending, prefer longer cycles
            cycles.sort(key=len, reverse=True)

            added = False
            for cycle in cycles:
                cycle_edges = _cycle_edges(cycle)
                if cycle_edges.issubset(remaining) and not (cycle_edges & used_edges):
                    selected_cycles.append(cycle)
                    used_edges |= cycle_edges
                    remaining -= cycle_edges
                    for u, v in cycle_edges:
                        current_adj[u].discard(v)
                        current_adj[v].discard(u)
                    added = True
                    break

            if not added:
                break

        # Phase 2: Local search - try to merge or improve cycles
        for ls_iter in range(30):
            if time.time() > deadline - 0.3:
                break
            improved = False

            # Try to find a cycle that uses edges from two existing cycles
            # to create a longer cycle, reducing piece count
            for i in range(len(selected_cycles)):
                if time.time() > deadline - 0.3:
                    break
                for j in range(i + 1, len(selected_cycles)):
                    if time.time() > deadline - 0.3:
                        break
                    # Check if cycles i and j share a vertex
                    set_i = set(selected_cycles[i])
                    set_j = set(selected_cycles[j])
                    common_vertices = set_i & set_j
                    if not common_vertices:
                        continue

                    # Try to find a new cycle using edges from both
                    # Create a temporary graph with edges from both cycles
                    temp_edges = _cycle_edges(selected_cycles[i]) | _cycle_edges(selected_cycles[j])
                    temp_adj = [set() for _ in range(n)]
                    for u, v in temp_edges:
                        temp_adj[u].add(v)
                        temp_adj[v].add(u)

                    # Try to find a cycle in this temp graph
                    for v_start in common_vertices:
                        if len(temp_adj[v_start]) < 2:
                            continue
                        # DFS from v_start
                        visited = {v_start}
                        path = [v_start]
                        current = v_start
                        for _ in range(len(temp_edges)):
                            neighbors = list(temp_adj[current])
                            if v_start in neighbors and len(path) >= 3:
                                new_cycle = path[:]
                                new_cycle_edges = _cycle_edges(new_cycle)
                                # Check if this cycle uses edges from both original cycles
                                if new_cycle_edges.issubset(temp_edges):
                                    # Check if it's actually a merge (uses edges from both)
                                    edges_i = _cycle_edges(selected_cycles[i])
                                    edges_j = _cycle_edges(selected_cycles[j])
                                    if new_cycle_edges & edges_i and new_cycle_edges & edges_j:
                                        # Replace both cycles with this one
                                        remaining = remaining | _cycle_edges(selected_cycles[i]) | _cycle_edges(selected_cycles[j])
                                        remaining -= new_cycle_edges
                                        used_edges = used_edges - _cycle_edges(selected_cycles[i]) - _cycle_edges(selected_cycles[j])
                                        used_edges |= new_cycle_edges
                                        selected_cycles.pop(j)
                                        selected_cycles.pop(i)
                                        selected_cycles.append(new_cycle)
                                        # Update adjacency
                                        for u, v in _cycle_edges(selected_cycles[i]):
                                            current_adj[u].discard(v)
                                            current_adj[v].discard(u)
                                        improved = True
                                        break
                            unvisited = [v for v in neighbors if v not in visited]
                            if not unvisited:
                                break
                            next_v = rng_attempt.choice(unvisited)
                            path.append(next_v)
                            visited.add(next_v)
                            current = next_v
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        # Compute pieces
        pieces = [list(c) for c in selected_cycles]
        for e in remaining:
            pieces.append(list(e))

        cnt = len(pieces)
        if cnt < best_count:
            best_count = cnt
            best_pieces = pieces

    if best_pieces is None:
        best_pieces = [list(e) for e in edges]

    return best_pieces


def construct():
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
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
