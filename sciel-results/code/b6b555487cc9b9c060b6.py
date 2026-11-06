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

def _find_longest_cycle_greedy(adj, n, seed, deadline=None):
    """
    Find a long cycle using greedy longest-path extension.
    Returns a list of vertices forming a cycle, or empty list if none found.
    """
    if not adj:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    rng.shuffle(vertices)

    # Try multiple starting vertices (top by degree + random)
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:5] + vertices[:5]))
    rng.shuffle(starts)

    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # Greedy longest path from start: always extend to unvisited neighbor
        # with highest degree (keeps options open for closing the cycle)
        path = [start]
        visited = {start}

        while True:
            u = path[-1]
            candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
            if not candidates:
                break
            candidates.sort(reverse=True)
            # Randomize among top-3 to add diversity
            top_k = min(3, len(candidates))
            _, next_v = rng.choice(candidates[:top_k])
            path.append(next_v)
            visited.add(next_v)

        # Check if we can close the cycle directly
        if len(path) >= 3 and start in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)

        # Look for any back-edge (path[i], path[j]) with i < j-1 to form a cycle
        # Take the longest such cycle
        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(len(path) - 1, i + 1, -1):
                if path[i] in adj[path[j]]:
                    cycle = path[i:j + 1]
                    if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break  # Found the longest cycle starting at i

    return best_cycle

def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []

    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    global_deadline = start_time + time_limit

    # Initial greedy decomposition to have a baseline
    def greedy_decompose(remaining_edges, n, seed_offset=0):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(list)
        for u, v in remaining:
            adj[u].append(v)
            adj[v].append(u)

        cycle_counter = 0
        while remaining:
            # Find a long cycle
            cycle = _find_longest_cycle_greedy(adj, n, seed + seed_offset + cycle_counter, deadline=global_deadline)
            if cycle:
                # Extract cycle edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)

                # Remove from remaining and adj
                for e in cycle_edges:
                    if e in remaining:
                        u, v = e
                        _remove_edge(adj, remaining, u, v)

                pieces.append(cycle)
                cycle_counter += 1
            else:
                # No cycle found, take a single edge
                # Pick an edge incident to a high-degree vertex to keep graph connected?
                # Or just any edge.
                # To maximize future cycles, maybe remove edges that are "bridges" or in small components?
                # Simple: pick random edge
                e = next(iter(remaining))
                u, v = e
                _remove_edge(adj, remaining, u, v)
                pieces.append([u, v])

        return pieces

    # Multi-start search
    # We use a larger number of restarts because the greedy method is fast
    for restart in range(500):
        if time.time() > global_deadline:
            break

        pieces = greedy_decompose(edges, n, restart)
        current_count = len(pieces)

        if current_count < best_count:
            best_count = current_count
            best_pieces = pieces

    if best_pieces is None:
        # Fallback: all edges as single pieces
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=40.0)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
