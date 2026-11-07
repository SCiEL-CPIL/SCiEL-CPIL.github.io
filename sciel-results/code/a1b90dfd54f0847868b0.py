"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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


def _find_hamiltonian_fast(adj, adj_mask, used, deadline, start=0):
    """Optimized DFS with bitmask adjacency."""
    n = P['n']
    calls = [0]

    def dfs(path, pos_mask, count):
        calls[0] += 1
        if calls[0] & 0x3FF == 0:
            if time.time() > deadline:
                return None
        if count == n:
            last = path[-1]
            e = _edge_key(last, start)
            if e not in used and (adj_mask[last] >> start) & 1:
                return path[:]
            return None
        last = path[-1]
        avail = adj_mask[last] & ~pos_mask
        if count < n - 1:
            avail &= ~(1 << start)
        cands = []
        while avail:
            lsb = avail & (-avail)
            y = lsb.bit_length() - 1
            if _edge_key(last, y) not in used:
                cands.append(y)
            avail ^= lsb
        if not cands:
            return None
        def fwd(y):
            cnt = 0
            m2 = adj_mask[y] & ~pos_mask & ~(1 << start)
            while m2:
                lsb2 = m2 & (-m2)
                z = lsb2.bit_length() - 1
                if _edge_key(y, z) not in used:
                    cnt += 1
                m2 ^= lsb2
            return cnt
        cands.sort(key=fwd)
        for y in cands:
            path.append(y)
            pos_mask |= (1 << y)
            res = dfs(path, pos_mask, count + 1)
            if res is not None:
                return res
            pos_mask ^= (1 << y)
            path.pop()
        return None

    path = [start]
    pos_mask = 1 << start
    return dfs(path, pos_mask, 1)


def _try_random_prefix_cycle(adj, adj_mask, used, deadline, start, rng):
    """Find Hamiltonian cycle with random prefix to diversify search."""
    n = P['n']
    prefix = [start]
    pos_mask = 1 << start
    remaining = list(range(1, n))
    rng.shuffle(remaining)
    steps = rng.randint(3, 7)
    for _ in range(steps):
        if not remaining:
            break
        last = prefix[-1]
        options = [v for v in remaining if (adj_mask[last] >> v) & 1 and _edge_key(last, v) not in used]
        if not options:
            return None
        v = rng.choice(options)
        prefix.append(v)
        pos_mask |= (1 << v)
        remaining.remove(v)
    if len(remaining) == 0:
        last = prefix[-1]
        if (adj_mask[last] >> start) & 1 and _edge_key(last, start) not in used:
            return prefix[:]
        return None
    calls = [0]
    def dfs(path, pm, count):
        calls[0] += 1
        if calls[0] & 0x1FF == 0:
            if time.time() > deadline:
                return None
        if count == n:
            last = path[-1]
            e = _edge_key(last, start)
            if e not in used and (adj_mask[last] >> start) & 1:
                return path[:]
            return None
        last = path[-1]
        avail = adj_mask[last] & ~pm
        if count < n - 1:
            avail &= ~(1 << start)
        cands = []
        while avail:
            lsb = avail & (-avail)
            y = lsb.bit_length() - 1
            if _edge_key(last, y) not in used:
                cands.append(y)
            avail ^= lsb
        if not cands:
            return None
        def fwd(y):
            cnt = 0
            m2 = adj_mask[y] & ~pm & ~(1 << start)
            while m2:
                lsb2 = m2 & (-m2)
                z = lsb2.bit_length() - 1
                if _edge_key(y, z) not in used:
                    cnt += 1
                m2 ^= lsb2
            return cnt
        cands.sort(key=fwd)
        for y in cands:
            path.append(y)
            pm |= (1 << y)
            res = dfs(path, pm, count + 1)
            if res is not None:
                return res
            pm ^= (1 << y)
            path.pop()
        return None
    return dfs(prefix, pos_mask, len(prefix))


def construct():
    deadline = time.time() + 80
    n = P['n']
    adj = {u: set() for u in range(n)}
    for (u, v) in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    adj_mask = [0] * n
    for (u, v) in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    vertex_set = set(range(n))
    used = set()
    cycles = []
    rng = random.Random(12345)

    # Cycle 1: natural path
    c1 = list(range(n))
    if _is_valid_cycle(c1, adj, vertex_set):
        cycles.append(c1)
        for e in _cycle_edges(c1):
            used.add(e)
    else:
        c1 = _find_hamiltonian_fast(adj, adj_mask, used, deadline)
        if c1 is not None and _is_valid_cycle(c1, adj, vertex_set):
            cycles.append(c1)
            for e in _cycle_edges(c1):
                used.add(e)

    # Cycle 2
    if time.time() < deadline:
        c2 = None
        patterns = [
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1],
            [0, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 2],
            [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2],
        ]
        for pat in patterns:
            if _is_valid_cycle(pat, adj, vertex_set) and all(e not in used for e in _cycle_edges(pat)):
                c2 = pat
                break
        if c2 is None:
            c2 = _find_hamiltonian_fast(adj, adj_mask, used, deadline)
        if c2 is not None and _is_valid_cycle(c2, adj, vertex_set) and all(e not in used for e in _cycle_edges(c2)):
            cycles.append(c2)
            for e in _cycle_edges(c2):
                used.add(e)

    # Cycle 3
    if time.time() < deadline:
        c3 = None
        c3 = _find_hamiltonian_fast(adj, adj_mask, used, deadline)
        if c3 is None:
            remaining_time = deadline - time.time()
            attempts = min(30, max(1, int(remaining_time * 3)))
            for attempt in range(attempts):
                if time.time() > deadline:
                    break
                start = rng.randint(0, n - 1)
                c3 = _try_random_prefix_cycle(adj, adj_mask, used, deadline, start, rng)
                if c3 is not None:
                    break
        if c3 is not None and _is_valid_cycle(c3, adj, vertex_set) and all(e not in used for e in _cycle_edges(c3)):
            cycles.append(c3)
            for e in _cycle_edges(c3):
                used.add(e)

    # Validate
    while cycles and not _is_valid_solution(cycles, adj, vertex_set):
        cycles.pop()

    if not cycles:
        fb = _find_hamiltonian_fast(adj, adj_mask, set(), deadline)
        if fb is None:
            fb = list(range(n))
        return [[int(v) for v in fb]]

    return [[int(v) for v in c] for c in cycles]
# EVOLVE-BLOCK-END
