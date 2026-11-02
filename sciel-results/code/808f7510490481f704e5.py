"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _next_prime(n):
    if n <= 2:
        return 2
    c = n if n % 2 else n + 1
    while not _is_prime(c):
        c += 2
    return c

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
    q = _next_prime(k)
    d = g - 5 if (g - 5) % 2 else g - 4
    d = max(3, d)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)

    n = len(order)
    adj = [[] for _ in range(n)]
    for v_idx, vertex in enumerate(order):
        if vertex[0] != 0:
            continue
        for other in neighbours(vertex):
            o_idx = index[other]
            adj[v_idx].append(o_idx)
            adj[o_idx].append(v_idx)
    return n, adj

def _find_short_cycle(n, adj, g):
    """Return a list of vertices forming a cycle of length < g, or None."""
    for start in range(n):
        # BFS from start, tracking parent to detect cycles
        # We look for any cycle of length < g using non-backtracking BFS
        visited = {}  # (node, prev) -> (dist, path)
        for nb in adj[start]:
            visited[(nb, start)] = (1, [start, nb])
            if len(adj[nb]) == 1:
                continue
        # BFS layer by layer
        frontier = list(visited.keys())
        depth = 1
        while frontier and depth < g - 1:
            next_frontier = []
            for (u, prev) in frontier:
                for w in adj[u]:
                    if w == prev:
                        continue
                    if w == start:
                        # Found a cycle: path from (u,prev) back to start
                        path = visited[(u, prev)][1]
                        return path  # cycle of length len(path)
                    key = (w, u)
                    if key not in visited:
                        visited[key] = (depth + 1, visited[(u, prev)][1] + [w])
                        next_frontier.append(key)
                    else:
                        # Back-edge to an earlier vertex (not start)
                        # This means there's a cycle
                        old_path = visited[key][1]
                        new_path = visited[(u, prev)][1] + [w]
                        # The cycle is from w's first occurrence in old_path to end of new_path
                        try:
                            w_idx = old_path.index(w)
                        except ValueError:
                            continue
                        cycle = old_path[w_idx:] + new_path[w_idx:]
                        if len(cycle) < g:
                            return cycle
            frontier = next_frontier
            depth += 1
    return None

def _try_2switch(n, adj, g, deadline):
    """Try to eliminate short cycles via random 2-switches."""
    # Build edge set
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))

    def add_edge(u, v):
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u, v), max(u, v)))

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        edge_set.discard((min(u, v), max(u, v)))

    # Find all edges
    edges = sorted(edge_set)

    # Try random 2-switches to break short cycles
    rng = random.Random(42)
    max_attempts = 50000

    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        # Find a short cycle
        cycle = _find_short_cycle(n, adj, g)
        if cycle is None:
            break  # No short cycles!
        
        # Pick two non-adjacent edges in the cycle to swap
        clen = len(cycle)
        if clen < 4:
            break  # Triangle can't be fixed by 2-switch easily
        
        # Try several random pairs of edges in the cycle
        found = False
        for _ in range(20):
            if time.time() > deadline:
                break
            i = rng.randint(0, clen - 1)
            j = rng.randint(0, clen - 1)
            if abs(i - j) <= 1 or abs(i - j) >= clen - 1:
                continue  # adjacent edges
            a1 = cycle[i]
            b1 = cycle[(i + 1) % clen]
            a2 = cycle[j]
            b2 = cycle[(j + 1) % clen]
            # Remove edges (a1,b1) and (a2,b2), add (a1,a2) and (b1,b2)
            if (a1, b1) not in edge_set or (a2, b2) not in edge_set:
                continue
            if (a1, a2) in edge_set or (b1, b2) in edge_set:
                continue  # would create duplicate
            remove_edge(a1, b1)
            remove_edge(a2, b2)
            add_edge(a1, a2)
            add_edge(b1, b2)
            found = True
            break
        if not found:
            # Try random 2-switch anywhere
            if len(edges) < 4:
                break
            e1 = rng.choice(list(edge_set))
            e2 = rng.choice(list(edge_set))
            if e1 == e2:
                continue
            a1, b1 = e1
            a2, b2 = e2
            if len({a1, b1, a2, b2}) < 4:
                continue
            if (a1, a2) in edge_set or (b1, b2) in edge_set:
                continue
            remove_edge(a1, b1)
            remove_edge(a2, b2)
            add_edge(a1, a2)
            add_edge(b1, b2)

    return adj

def construct(k, g):
    deadline = time.time() + 100  # 100s safety margin
    
    # Step 1: Construct LUW graph
    n, adj = _luw_construct(k, g)
    
    # Step 2: Check for short cycles and repair if needed
    if time.time() < deadline:
        cycle = _find_short_cycle(n, adj, g)
        if cycle is not None:
            adj = _try_2switch(n, adj, g, deadline)
    
    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    return n, edges

# EVOLVE-BLOCK-END
