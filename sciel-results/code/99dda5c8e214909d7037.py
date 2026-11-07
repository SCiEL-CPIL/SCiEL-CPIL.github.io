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


def _find_hamiltonian(adj, edge_set, used, deadline, start=0):
    """DFS for a Hamiltonian cycle on the pinned vertex set, avoiding used edges."""
    n = P['n']
    # Precompute the full neighbor list per vertex (only host edges matter;
    # used edges are filtered dynamically inside the search).
    nbrs = [sorted(adj[v]) for v in range(n)]

    path = [start]
    pos = [False] * n
    pos[start] = True
    count = 1
    # remaining unvisited count (excluding start)
    remaining = n - 1

    # For pruning: precompute, for each vertex, the set of unvisited-eligible
    # neighbours is dynamic; instead use a cheap connectivity check below.
    def connected_from(cur):
        # BFS over unused edges among unvisited vertices (plus cur as entry)
        # to ensure every unvisited vertex is reachable from cur via unused edges.
        # This is a strong but somewhat costly prune; only call when remaining is small.
        seen = {cur}
        stack = [cur]
        target = 0
        while stack:
            x = stack.pop()
            for y in nbrs[x]:
                if y in seen:
                    continue
                if y == start:
                    continue
                if not pos[y]:  # unvisited
                    target += 1
                    seen.add(y)
                    stack.append(y)
                else:
                    # y already visited (not start): cannot traverse through it
                    continue
        # target counts unvisited vertices reachable; must equal remaining
        return target == remaining

    def dfs():
        nonlocal count, remaining
        if time.time() > deadline:
            return None
        if count == n:
            last = path[-1]
            e = _edge_key(last, start)
            if e not in used and last in adj[start]:
                return path[:]
            return None
        last = path[-1]
        # Order neighbours: prefer those with few unused options (MRV)
        cands = []
        for y in nbrs[last]:
            if y == start:
                continue
            if pos[y]:
                continue
            if _edge_key(last, y) in used:
                continue
            cands.append(y)
        # MRV: sort by number of forward options (unused, unvisited)
        def fwd(y):
            k = 0
            for z in nbrs[y]:
                if z == start:
                    continue
                if pos[z]:
                    continue
                if _edge_key(y, z) in used:
                    continue
                k += 1
            return k
        cands.sort(key=fwd)
        for y in cands:
            path.append(y)
            pos[y] = True
            count += 1
            remaining -= 1
            res = dfs()
            if res is not None:
                return res
            # backtrack
            remaining += 1
            count -= 1
            pos[y] = False
            path.pop()
        return None

    return dfs()


def _complete_from_residual(adj, edge_set, used):
    """If residual (host edges minus used) is a single 2-regular Hamiltonian cycle,
    walk it and return the vertex list. Returns None otherwise."""
    n = P['n']
    residual = set()
    for (u, v) in P['edges']:
        e = _edge_key(u, v)
        if e not in used:
            residual.add(e)
    rem = [[] for _ in range(n)]
    for (u, v) in residual:
        rem[u].append(v)
        rem[v].append(u)
    for u in range(n):
        if len(rem[u]) != 2:
            return None
    # walk
    cyc = [0]
    prev = -1
    cur = 0
    for _ in range(n):
        a, b = rem[cur]
        nxt = a if a != prev else b
        if nxt == 0 and len(cyc) < n:
            return None
        cyc.append(nxt)
        prev, cur = cur, nxt
        if cur == 0:
            break
    if len(cyc) != n + 1 or cyc[-1] != 0:
        return None
    return cyc[1:]


def construct():
    deadline = time.time() + 80  # safety margin under 90s kill

    n = P['n']
    adj = {u: set() for u in range(n)}
    edge_set = set()
    for (u, v) in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(_edge_key(u, v))

    vertex_set = set(range(n))

    # Known good first cycle: the natural path 0..20 (its edges are all host edges).
    base = list(range(n))
    if not _is_valid_cycle(base, adj, vertex_set):
        base = None

    used = set()
    cycles = []

    # Cycle 1: try the natural path first (instant), else DFS.
    if base is not None:
        c1 = base
    else:
        c1 = _find_hamiltonian(adj, edge_set, used, deadline)
    if c1 is not None and _is_valid_cycle(c1, adj, vertex_set):
        cycles.append(c1)
        for e in _cycle_edges(c1):
            used.add(e)

    # Cycle 2: DFS on residual.
    if time.time() < deadline:
        c2 = _find_hamiltonian(adj, edge_set, used, deadline)
        if c2 is not None and _is_valid_cycle(c2, adj, vertex_set):
            cycles.append(c2)
            for e in _cycle_edges(c2):
                used.add(e)

    # Cycle 3: try analytic residual completion, then DFS.
    if time.time() < deadline:
        c3 = _complete_from_residual(adj, edge_set, used)
        if c3 is None or not _is_valid_cycle(c3, adj, vertex_set):
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
