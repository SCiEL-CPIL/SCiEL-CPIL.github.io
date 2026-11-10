"""Packing edge-disjoint Hamiltonian cycles on a 21-vertex host."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(100000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
ALL_EDGES = set()
for a, b in P['edges']:
    ALL_EDGES.add((a, b) if a < b else (b, a))

# Build adjacency as list of neighbor lists
ADJ = [set() for _ in range(N)]
for a, b in ALL_EDGES:
    ADJ[a].add(b)
    ADJ[b].add(a)

DEADLINE = time.time() + 55.0  # safety margin before 60s kill / 90s wall


def _key(e):
    a, b = e
    return (a, b) if a < b else (b, a)


def _find_hamiltonian(adj, start, end, used_mask, time_budget):
    """Find a Hamiltonian cycle from start to end visiting all N vertices,
    using only edges NOT in used_mask. adj is list of sets.
    Returns list of vertices (cycle order) or None.
    """
    n = N
    # Precompute for each vertex the list of neighbors, sorted by degree (ascending) for pruning
    # We do a DFS with backtracking.
    visited = [False] * n
    path = [start]
    visited[start] = True
    # Order of neighbors: try lower-degree first (heuristic)
    neigh_order = []
    for v in range(n):
        nbrs = list(adj[v])
        # sort by degree ascending (more constrained first)
        nbrs.sort(key=lambda x: len(adj[x]))
        neigh_order.append(nbrs)

    # Special handling: end must be reachable and we must close the cycle.
    # We require that the last vertex in path connects to end, and end connects to start.
    # Actually the cycle is path[0], path[1], ..., path[n-1], then back to path[0].
    # So we need edge (path[n-1], path[0]) to exist and be unused.
    # And we need to visit 'end'? No - we need to visit ALL vertices.
    # Let me reconsider: we just need a Hamiltonian cycle. Start is fixed (0 for symmetry).
    # The cycle visits all vertices and returns to start.

    found = [None]

    def dfs(u, depth):
        if time.time() > time_budget:
            return
        if depth == n - 1:
            # Check if u connects back to start with unused edge
            if start in adj[u] and _key((u, start)) not in used_mask:
                found[0] = list(path)
                return True
            return False
        for v in neigh_order[u]:
            if visited[v]:
                continue
            if _key((u, v)) in used_mask:
                continue
            # Pruning: if v == start and depth < n-1, skip (can't revisit start)
            # start is already visited so this is handled
            visited[v] = True
            path.append(v)
            if dfs(v, depth + 1):
                return True
            path.pop()
            visited[v] = False
        return False

    if dfs(start, 0):
        return found[0]
    return None


def _find_hamiltonian_greedy(adj, used_mask, start, time_budget, order_seed=0):
    """Try to find a Hamiltonian cycle with a more guided approach.
    Returns list of vertices or None."""
    n = N
    visited = [False] * n
    path = [start]
    visited[start] = True

    # Build neighbor lists with ordering
    import random
    rng = random.Random(order_seed)
    neigh_order = []
    for v in range(n):
        nbrs = list(adj[v])
        # Mix degree ordering with randomization
        nbrs.sort(key=lambda x: (len(adj[x]), rng.random()))
        neigh_order.append(nbrs)

    found = [None]

    def dfs(u, depth):
        if time.time() > time_budget:
            return
        if depth == n - 1:
            if start in adj[u] and _key((u, start)) not in used_mask:
                found[0] = list(path)
                return True
            return False
        for v in neigh_order[u]:
            if visited[v]:
                continue
            if _key((u, v)) in used_mask:
                continue
            visited[v] = True
            path.append(v)
            if dfs(v, depth + 1):
                return True
            path.pop()
            visited[v] = False
        return False

    if dfs(start, 0):
        return found[0]
    return None


def _build_adj_from_remaining(remaining_edges):
    """Build adjacency sets from a set of remaining edges."""
    adj = [set() for _ in range(N)]
    for a, b in remaining_edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _validate_cycle(cycle, remaining_edges):
    """Check that cycle is a valid Hamiltonian cycle using only remaining edges."""
    if len(cycle) != N:
        return False
    if len(set(cycle)) != N:
        return False
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        e = _key((u, v))
        if e not in remaining_edges:
            return False
    return True


def _try_find_k_cycles(k, time_budget):
    """Try to find k edge-disjoint Hamiltonian cycles. Returns list of cycles or None."""
    remaining = set(ALL_EDGES)
    cycles = []
    for i in range(k):
        if time.time() > time_budget:
            break
        adj = _build_adj_from_remaining(remaining)
        # Check each vertex has degree >= 2 in remaining graph
        ok = True
        for v in range(N):
            if len(adj[v]) < 2:
                ok = False
                break
        if not ok:
            break
        cycle = _find_hamiltonian(adj, 0, 0, set(), time_budget)
        if cycle is None:
            # Try a few different orderings
            for seed in range(5):
                if time.time() > time_budget:
                    break
                cycle = _find_hamiltonian_greedy(adj, set(), 0, time_budget, seed)
                if cycle is not None:
                    break
        if cycle is None:
            break
        # Validate
        if not _validate_cycle(cycle, remaining):
            break
        cycles.append(cycle)
        # Remove edges
        for j in range(N):
            u = cycle[j]
            v = cycle[(j + 1) % N]
            e = _key((u, v))
            remaining.discard(e)
    return cycles if len(cycles) == k else cycles


def construct():
    start_time = time.time()
    budget = 55.0

    # Try to find 3 cycles first
    best = []
    for k in [3, 2, 1]:
        if time.time() - start_time > budget - 5:
            break
        cycles = _try_find_k_cycles(k, time.time() + min(10.0, budget - (time.time() - start_time)))
        if len(cycles) == k:
            best = cycles
            break
        elif len(cycles) > len(best):
            best = cycles

    # If we only got fewer, try harder with more time
    if len(best) < 3:
        remaining_time = budget - (time.time() - start_time)
        if remaining_time > 5:
            # Try again with fresh search, different ordering
            for attempt in range(3):
                if time.time() - start_time > budget - 2:
                    break
                remaining = set(ALL_EDGES)
                cycles = []
                for i in range(3):
                    if time.time() - start_time > budget - 2:
                        break
                    adj = _build_adj_from_remaining(remaining)
                    ok = all(len(adj[v]) >= 2 for v in range(N))
                    if not ok:
                        break
                    cycle = None
                    for seed in range(attempt * 3 + 1):
                        if time.time() - start_time > budget - 2:
                            break
                        cycle = _find_hamiltonian_greedy(adj, set(), 0, time.time() + 1.0, seed + attempt * 10)
                        if cycle is not None:
                            break
                    if cycle is None:
                        break
                    if not _validate_cycle(cycle, remaining):
                        break
                    cycles.append(cycle)
                    for j in range(N):
                        u = cycle[j]
                        v = cycle[(j + 1) % N]
                        e = _key((u, v))
                        remaining.discard(e)
                if len(cycles) > len(best):
                    best = cycles
                if len(best) == 3:
                    break

    # Return as list of lists of Python ints
    result = []
    for cycle in best:
        result.append([int(v) for v in cycle])

    # Ensure nonempty
    if not result:
        # Fallback: try one more time with very basic search
        adj = _build_adj_from_remaining(set(ALL_EDGES))
        cycle = _find_hamiltonian(adj, 0, 0, set(), time.time() + 2.0)
        if cycle is not None:
            result = [[int(v) for v in cycle]]
        else:
            # Absolute fallback: a simple cycle if we can find one
            # This shouldn't happen for this graph
            result = [[0, 1, 2]]  # will be invalid but must return nonempty

    return result
# EVOLVE-BLOCK-END
