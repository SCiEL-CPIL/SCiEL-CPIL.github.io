"""Rewritten program: local search with fast spectral radius for prism graph lifts."""
import time
import numpy as np


def _build_edge_map(graph):
    """Map each undirected edge (u,v) to its index, handling both orientations."""
    edges = graph['edges']
    n = graph['n']
    idx = {}
    for i, (u, v) in enumerate(edges):
        idx[(min(u, v), max(u, v))] = i
    return idx, edges


def _signed_matrix(signs, edges, n):
    """Build the signed adjacency matrix."""
    A = np.zeros((n, n), dtype=np.float64)
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        A[u, v] = s
        A[v, u] = s
    return A


def _spectral_radius(signs, edges, n):
    """Compute spectral radius via eigvalsh on the symmetric signed matrix."""
    A = _signed_matrix(signs, edges, n)
    ev = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(ev)))


def _is_connected_lift(signs, edges, n):
    """
    Check that the 2-lift is connected and 3-regular.
    2-lift vertices: (v, 0), (v, 1) for v in 0..n-1.
    Edge (u,v) with sign +1: connects (u,0)-(v,0) and (u,1)-(v,1).
    Edge (u,v) with sign -1: connects (u,0)-(v,1) and (u,1)-(v,0).
    Each vertex in the lift has degree 3 (since base is 3-regular), so 3-regular is automatic.
    We just need connectivity.
    """
    total = 2 * n
    adj = [[] for _ in range(total)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            adj[u * 2].append(v * 2)
            adj[v * 2].append(u * 2)
            adj[u * 2 + 1].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 1)
        else:
            adj[u * 2].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2)
            adj[u * 2 + 1].append(v * 2)
            adj[v * 2].append(u * 2 + 1)
    # BFS
    visited = [False] * total
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if not visited[y]:
                visited[y] = True
                count += 1
                stack.append(y)
    return count == total


def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy: Local search with multiple restarts to minimize spectral radius,
    with validity (connected lift) enforced.
    """
    deadline = time.time() + 85.0  # Safety margin (90s limit)

    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)

    best_signs = None
    best_radius = float('inf')

    def evaluate(signs):
        """Return (radius, valid)."""
        if not _is_connected_lift(signs, edges, n):
            return float('inf'), False
        r = _spectral_radius(signs, edges, n)
        return r, True

    def local_search(start_signs, max_iters=2000):
        """Hill-climbing / first-improvement local search."""
        nonlocal best_signs, best_radius
        signs = start_signs.copy()
        cur_radius, valid = evaluate(signs)
        if not valid:
            # Try to fix validity by random flips until valid
            for _ in range(200):
                signs = start_signs.copy()
                cur_radius, valid = evaluate(signs)
                if valid:
                    break
            if not valid:
                return

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for i in range(num_edges):
                if time.time() >= deadline:
                    break
                signs[i] = -signs[i]
                new_radius, valid = evaluate(signs)
                if valid and new_radius < cur_radius:
                    cur_radius = new_radius
                    improved = True
                    if cur_radius < best_radius:
                        best_radius = cur_radius
                        best_signs = signs.copy()
                else:
                    signs[i] = -signs[i]

    # Generate random valid starting points
    rng = np.random.default_rng(3)

    # First, get a baseline valid configuration
    # Start with all +1 and check; if invalid, flip edges until valid
    base_signs = np.ones(num_edges, dtype=np.int8)
    _, valid = evaluate(base_signs)
    if not valid:
        # Randomly flip until we find a valid one
        for attempt in range(500):
            trial = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            _, valid = evaluate(trial)
            if valid:
                base_signs = trial
                break

    # Run local search from the base
    local_search(base_signs, max_iters=3000)

    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        # Random start, but bias toward valid
        trial = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        _, valid = evaluate(trial)
        if not valid:
            # Try a few flips to get valid
            for _ in range(50):
                flip_idx = rng.integers(0, num_edges)
                trial[flip_idx] = -trial[flip_idx]
                _, valid = evaluate(trial)
                if valid:
                    break
            if not valid:
                continue
        local_search(trial, max_iters=1500)

    if best_signs is None:
        # Fallback: return the base (should be valid)
        return base_signs
    return best_signs
