"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def _edge_key(u, v):
    return (min(u, v), max(u, v))


def _cycle_edges(cycle):
    m = len(cycle)
    out = []
    for i in range(m):
        out.append(_edge_key(cycle[i], cycle[(i + 1) % m]))
    return out


def _is_valid_cycle(cycle, adj, vertex_set):
    if not isinstance(cycle, (list, tuple)):
        return False
    m = len(cycle)
    if m < 3:
        return False
    if len(cycle) != len(set(cycle)):
        return False
    for v in cycle:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v not in vertex_set:
            return False
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        if b not in adj[a]:
            return False
    if set(cycle) != vertex_set:
        return False
    return True


def _is_valid_solution(cycles, adj, vertex_set):
    for c in cycles:
        if not _is_valid_cycle(c, adj, vertex_set):
            return False
    seen = set()
    for c in cycles:
        for e in _cycle_edges(c):
            if e in seen:
                return False
            seen.add(e)
    return True


def _find_hamiltonian(adj, edge_set, used, deadline):
    """DFS for a Hamiltonian cycle on the pinned vertex set, avoiding used edges.
    Uses connectivity pruning and MRV ordering."""
    n = P['n']
    nbrs = [sorted(adj[v]) for v in range(n)]
    start = 0

    # Precompute adjacency as sets for fast lookup
    adj_set = [set(nbrs[v]) for v in range(n)]

    def has_path_from(cur, unvisited_set, remaining):
        """Check if all unvisited vertices are reachable from cur via unused edges."""
        if remaining == 0:
            return True
        # BFS from cur through unvisited vertices using unused edges
        visited = set()
        stack = [cur]
        visited.add(cur)
        while stack:
            x = stack.pop()
            for y in nbrs[x]:
                if y in visited:
                    continue
                if y == start:
                    continue
                if y in unvisited_set:
                    if _edge_key(x, y) not in used:
                        visited.add(y)
                        stack.append(y)
                # else: y is visited or start, skip
        # Count how many unvisited vertices we reached
        reached = len(visited) - 1  # exclude cur
        return reached >= remaining - 1  # we need all unvisited reachable

    def dfs():
        if time.time() > deadline:
            return None
        n_local = len(path)
        if n_local == n:
            last = path[-1]
            e = _edge_key(last, start)
            if e not in used and last in adj_set[start]:
                return path[:]
            return None
        last = path[-1]
        unvisited_set = set(range(n)) - set(path)
        remaining = n - n_local

        # Get candidates: unvisited neighbors of last with unused edge
        cands = []
        for y in nbrs[last]:
            if y == start:
                continue
            if y in unvisited_set:
                if _edge_key(last, y) not in used:
                    cands.append(y)

        if not cands:
            return None

        # MRV: sort by number of forward options
        def fwd_count(y):
            k = 0
            for z in nbrs[y]:
                if z == start:
                    continue
                if z in unvisited_set:
                    if _edge_key(y, z) not in used:
                        k += 1
            return k

        cands.sort(key=fwd_count)

        # Connectivity prune: check if remaining unvisited are reachable
        # Only do this when remaining is small to save time
        if remaining <= 5:
            # Quick connectivity check
            test_unvisited = unvisited_set - set(cands)
            if test_unvisited:
                # Check if all test_unvisited reachable from last via cands
                reachable = set()
                stack = list(cands)
                reachable.update(cands)
                while stack:
                    x = stack.pop()
                    for y in nbrs[x]:
                        if y in reachable or y == start:
                            continue
                        if y in test_unvisited and _edge_key(x, y) not in used:
                            reachable.add(y)
                            stack.append(y)
                if len(reachable & test_unvisited) < len(test_unvisited):
                    return None

        for y in cands:
            path.append(y)
            res = dfs()
            if res is not None:
                return res
            path.pop()
        return None

    path = [start]
    return dfs()


def construct():
    deadline = time.time() + 70  # safety margin under 90s kill

    n = P['n']
    adj = {u: set() for u in range(n)}
    edge_set = set()
    for (u, v) in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(_edge_key(u, v))

    vertex_set = set(range(n))

    # Known good first cycle: the natural path 0..20
    base = list(range(n))
    if not _is_valid_cycle(base, adj, vertex_set):
        base = None

    used = set()
    cycles = []

    # Cycle 1: try the natural path first (instant)
    if base is not None:
        c1 = base
    else:
        c1 = _find_hamiltonian(adj, edge_set, used, deadline)
    if c1 is not None and _is_valid_cycle(c1, adj, vertex_set):
        cycles.append(c1)
        for e in _cycle_edges(c1):
            used.add(e)

    # Cycle 2: DFS on residual
    if time.time() < deadline:
        c2 = _find_hamiltonian(adj, edge_set, used, deadline)
        if c2 is not None and _is_valid_cycle(c2, adj, vertex_set):
            cycles.append(c2)
            for e in _cycle_edges(c2):
                used.add(e)

    # Cycle 3: DFS on residual
    if time.time() < deadline:
        c3 = _find_hamiltonian(adj, edge_set, used, deadline)
        if c3 is not None and _is_valid_cycle(c3, adj, vertex_set):
            cycles.append(c3)

    # Validate the full set; drop invalid trailing cycles until valid.
    while cycles and not _is_valid_solution(cycles, adj, vertex_set):
        cycles.pop()

    if not cycles:
        # Absolute fallback: a single valid Hamiltonian cycle.
        fb = _find_hamiltonian(adj, edge_set, set(), deadline)
        if fb is None:
            fb = list(range(n))
        return [[int(v) for v in fb]]

    return [[int(v) for v in c] for c in cycles]
# EVOLVE-BLOCK-END
