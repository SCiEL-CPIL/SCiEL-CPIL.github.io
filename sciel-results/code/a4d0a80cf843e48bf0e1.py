"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict

# Hard deadline guard (evaluator kills at 150s; use 120s budget with safety margin)
_DEADLINE_SECONDS = 110.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(edges, n):
    """Build adjacency list and degree counts."""
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg


def _find_cycle(adj, deg, n):
    """
    Find a cycle in the current graph using a heuristic walk.
    Returns a list of vertices forming a cycle (length >= 3), or None.
    Uses a greedy approach: start from a vertex, follow unused edges,
    and try to close back to the start.
    """
    # Try to find a cycle by DFS from each vertex with odd degree or high degree
    # First, try vertices with degree >= 2
    candidates = [v for v in range(n) if deg[v] >= 2]
    if not candidates:
        return None

    # Sort by degree descending for better chances
    candidates.sort(key=lambda v: -deg[v])

    for start in candidates:
        if deg[start] < 2:
            continue
        # DFS to find a cycle back to start
        stack = [(start, [start])]
        visited = {start}
        found = False
        while stack:
            node, path = stack.pop()
            if found:
                break
            # Try neighbors in order of degree
            neighbors = sorted(adj[node], key=lambda x: -deg[x])
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    # Found a cycle
                    return path
                if nb not in visited:
                    visited.add(nb)
                    stack.append((nb, path + [nb]))
        # If no cycle found from this start via simple DFS, try a different approach

    # Fallback: try to find any cycle using a different method
    # Pick a random edge and try to extend it into a cycle
    all_edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                all_edges.add((u, v))
    if not all_edges:
        return None

    # Try each edge as a potential part of a cycle
    edge_list = list(all_edges)
    random.shuffle(edge_list)
    for u, v in edge_list:
        # Try to find a path from v back to u of length >= 2
        # BFS from v to u, avoiding the direct edge
        from collections import deque
        queue = deque()
        queue.append((v, [v]))
        visited = {v}
        while queue:
            node, path = queue.popleft()
            if node == u and len(path) >= 2:
                return path + [u]
            for nb in sorted(adj[node], key=lambda x: -deg[x]):
                if nb == u and len(path) >= 2:
                    return path + [u]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))

    return None


def _remove_cycle(cycle, adj, deg):
    """Remove edges of the cycle from the graph."""
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1


def _remove_edge(u, v, adj, deg):
    """Remove a single edge from the graph."""
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1


def _decompose_graph(edges, n, time_limit):
    """
    Decompose a graph into cycles and edges.
    Returns a list of pieces (each piece is either [u,v] for an edge or a list of >=3 vertices for a cycle).
    """
    adj, deg = _build_adjacency(edges, n)
    pieces = []
    start_time = time.time()

    # Greedily find cycles while possible
    while True:
        if time.time() - start_time > time_limit:
            break

        # Check if any edges remain
        remaining_edges = 0
        for u in range(n):
            remaining_edges += len(adj[u])
        remaining_edges //= 2

        if remaining_edges == 0:
            break

        # Try to find a cycle
        cycle = _find_cycle(adj, deg, n)
        if cycle is not None:
            pieces.append(cycle)
            _remove_cycle(cycle, adj, deg)
        else:
            # No cycle found; take an edge
            # Find any remaining edge
            edge_found = None
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    edge_found = (u, v)
                    break
            if edge_found is None:
                break
            u, v = edge_found
            pieces.append([u, v])
            _remove_edge(u, v, adj, deg)

    # Add any remaining edges as individual pieces
    for u in range(n):
        for v in adj[u]:
            if u < v:
                pieces.append([u, v])

    return pieces


def _decompose_with_restart(edges, n, time_limit, num_restarts=5):
    """
    Try multiple random restarts with different orderings to find a better decomposition.
    """
    best_pieces = None
    best_count = float('inf')
    per_restart_time = time_limit / max(num_restarts, 1)

    for restart in range(num_restarts):
        if time.time() > time_limit:
            break

        # Shuffle the adjacency ordering by randomizing tie-breaks
        # We'll do this by shuffling the order in which we consider vertices/edges
        pieces = _decompose_graph(edges, n, per_restart_time)
        count = len(pieces)

        if count < best_count:
            best_count = count
            best_pieces = pieces

    if best_pieces is None:
        # Fallback: all edges as individual pieces
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            # Not enough time; return all edges as pieces for remaining graphs
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)

        # Allocate time for this graph
        graph_time = remaining * 0.9  # Leave 10% buffer

        pieces = _decompose_with_restart(edges, n, graph_time, num_restarts=3)
        results.append(pieces)

    return results
