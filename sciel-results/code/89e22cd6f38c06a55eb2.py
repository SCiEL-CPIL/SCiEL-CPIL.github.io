import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_long_cycle(adj, n, rng, max_starts=None, max_restarts=6):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a randomized greedy strategy with backtracking.
    Returns the cycle as a list of vertices or None.
    """
    if max_starts is None:
        max_starts = min(n, 25)

    best_cycle = None
    best_len = 0

    # Try starting from each vertex with some randomization
    num_starts = min(n, max_starts)
    if n <= max_starts:
        starts = list(range(n))
    else:
        # Prioritize high-degree vertices
        degrees = [len(adj[v]) for v in range(n)]
        starts = rng.sample(range(n), num_starts)

    for start in starts:
        if not adj[start]:
            continue

        for _ in range(max_restarts):
            path = [start]
            visited = {start}
            current = start

            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]

                if not unvisited:
                    # Check if we can close the cycle
                    if start in adj[current] and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    # Heuristic: prefer neighbors with more unvisited connections
                    # Also consider closing early if path is already long
                    if len(path) >= 4 and start in adj[current] and rng.random() < 0.3:
                        # Close the cycle
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break

                    # Score each unvisited neighbor
                    scored = []
                    for v in unvisited:
                        # Count unvisited neighbors of v
                        unvisited_neighbors = sum(1 for w in adj[v] if w not in visited)
                        score = unvisited_neighbors
                        # Slight preference for closing to start if possible
                        if v == start:
                            score += 10
                        scored.append((score, v))

                    # Pick the best or randomly from top few
                    scored.sort(reverse=True)
                    top_k = min(3, len(scored))
                    if rng.random() < 0.7:
                        best_next = scored[0][1]
                    else:
                        best_next = rng.choice([s[1] for s in scored[:top_k]])

                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next

    return best_cycle


def _find_long_cycle_low_degree(adj, n, rng, max_starts=None, max_restarts=6):
    """
    Find a long simple cycle preferring low-degree neighbors to keep options open.
    Returns the cycle as a list of vertices or None.
    """
    if max_starts is None:
        max_starts = min(n, 25)

    best_cycle = None
    best_len = 0

    num_starts = min(n, max_starts)
    if n <= max_starts:
        starts = list(range(n))
    else:
        # Prioritize high-degree vertices for starting
        degrees = [len(adj[v]) for v in range(n)]
        starts = rng.sample(range(n), num_starts)

    for start in starts:
        if not adj[start]:
            continue

        for _ in range(max_restarts):
            path = [start]
            visited = {start}
            current = start

            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]

                if not unvisited:
                    if start in adj[current] and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    if len(path) >= 4 and start in adj[current] and rng.random() < 0.2:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                        break

                    # Prefer low-degree unvisited neighbors
                    if len(unvisited) > 1:
                        unvisited.sort(key=lambda v: len(adj[v]))
                        idx = rng.randrange(min(3, len(unvisited)))
                        next_node = unvisited[idx]
                    else:
                        next_node = unvisited[0]

                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

    return best_cycle


def _decompose_greedy(edges, n, deadline, rng=None, use_low_degree=False):
    """
    Greedy decomposition: repeatedly find a cycle and remove it.
    """
    if rng is None:
        rng = random.Random(12345)

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)

    while remaining > 0:
        if time.time() > deadline:
            break

        # Try to find a long cycle
        if use_low_degree:
            best_cycle = _find_long_cycle_low_degree(adj, n, rng, max_starts=min(n, 25), max_restarts=6)
        else:
            best_cycle = _find_long_cycle(adj, n, rng, max_starts=min(n, 25), max_restarts=6)

        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            # No cycle found, remove a single edge
            edge_found = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_found = True
                        break
                if edge_found:
                    break
            if not edge_found:
                break

    # Add any remaining edges as single edges
    if remaining > 0:
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    pieces.append([u, v])

    return pieces


def _polish_pieces(pieces, n, deadline):
    """
    Local search: try to merge adjacent edge pieces into cycles.
    """
    # Build edge set from edge pieces
    edge_set = set()
    for p in pieces:
        if len(p) == 2:
            u, v = p
            edge_set.add((min(u, v), max(u, v)))

    if not edge_set:
        return pieces

    # Find cycles in the edge-piece graph
    cycles = _find_cycles_in_edge_graph(edge_set, n, deadline, rng=random.Random(999))

    if not cycles:
        return pieces

    # Determine which edges are in found cycles
    used_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            used_edges.add((min(a, b), max(a, b)))

    # Rebuild pieces
    new_pieces = []
    for p in pieces:
        if len(p) >= 3:
            new_pieces.append(p)
        else:
            u, v = p
            e = (min(u, v), max(u, v))
            if e not in used_edges:
                new_pieces.append(p)

    # Add found cycles
    new_pieces.extend(cycles)

    return new_pieces


def _find_cycles_in_edge_graph(edge_set, n, deadline, rng=None):
    """
    Find cycles in the graph formed by edge_set.
    Returns a list of cycles (each cycle is a list of vertices).
    """
    if rng is None:
        rng = random.Random(999)

    if not edge_set:
        return []

    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    cycles = []
    used_edges = set()
    remaining = list(edge_set)

    max_rounds = 100
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if not remaining:
            break

        # Pick a random edge to start from
        start_edge = rng.choice(remaining)
        u, v = start_edge

        # Try to extend a cycle from this edge
        path = [u, v]
        visited = {u, v}
        current = v
        prev = u
        found = False

        for _ in range(100):  # Max path length
            if time.time() > deadline:
                break

            # Try to close the cycle
            if prev in adj[current] and len(path) >= 3:
                found = True
                break

            # Find unvisited neighbors
            unvisited = [x for x in adj[current] if x not in visited and x != prev]

            if not unvisited:
                break

            # Pick a random unvisited neighbor
            next_v = rng.choice(unvisited)
            path.append(next_v)
            visited.add(next_v)
            prev, current = current, next_v

        if found:
            # Extract the cycle edges
            cycle_edges = set()
            for i in range(len(path)):
                a = path[i]
                b = path[(i + 1) % len(path)]
                cycle_edges.add((min(a, b), max(a, b)))

            # Verify all edges are in edge_set and not already used
            if all(e in edge_set and e not in used_edges for e in cycle_edges):
                cycles.append(path)
                used_edges.update(cycle_edges)
                remaining = [e for e in remaining if e not in used_edges]

    return cycles


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')

    # Try multiple random strategies
    base_seed = 12345
    attempts = 0
    max_attempts = 200

    while time.time() < deadline - 1.0:
        seed = base_seed + attempts
        rng = random.Random(seed)
        
        # Alternate between high-degree and low-degree strategies
        use_low = (attempts % 2 == 1)
        pieces = _decompose_greedy(edges, n, deadline - 0.5, rng, use_low_degree=use_low)
        cnt = len(pieces)

        if cnt < best_count:
            best_count = cnt
            best = pieces[:]

        # Try polishing the best solution
        if best is not None and time.time() < deadline - 2.0:
            polished = _polish_pieces(best, n, deadline - 1.0)
            if len(polished) < best_count:
                best_count = len(polished)
                best = polished

        attempts += 1

        if attempts >= max_attempts:
            break

    return best if best else [[u, v] for u, v in edges]


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
