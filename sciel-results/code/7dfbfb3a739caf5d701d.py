"""Girth-increasing via 2-switch heuristic on random 3-regular graphs."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _random_3_regular(n, rng):
    """Generate a random 3-regular graph on n vertices via configuration model.
    Returns adjacency list (list of sets) or None if not simple."""
    if n < 4 or n % 2 != 0:
        return None
    # Create 3n half-edges: vertex i has half-edges (i,0),(i,1),(i,2)
    half_edges = []
    for i in range(n):
        for j in range(3):
            half_edges.append((i, j))
    # Shuffle
    rng.shuffle(half_edges)
    # Pair them up
    adj = [set() for _ in range(n)]
    for i in range(0, len(half_edges), 2):
        u, _ = half_edges[i]
        v, _ = half_edges[i + 1]
        if u == v:
            return None  # loop
        if v in adj[u]:
            return None  # multi-edge
        adj[u].add(v)
        adj[v].add(u)
    # Verify all degrees are 3
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return adj


def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g using BFS from each vertex.
    Returns (cycle_vertices, cycle_edges) or (None, None)."""
    for start in range(n):
        # BFS tracking parent
        parent = [-1] * n
        dist = [-1] * n
        parent[start] = start
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle: u-v-...-u
                    # Reconstruct
                    cycle = [u, v]
                    # Walk back from v to LCA
                    a, b = u, v
                    da, db = dist[u], dist[v]
                    # Find LCA
                    while a != b:
                        if da > db:
                            a = parent[a]
                            da -= 1
                        elif db > da:
                            b = parent[b]
                            db -= 1
                        else:
                            a = parent[a]
                            b = parent[b]
                            da -= 1
                            db -= 1
                    lca = a
                    # Path from u to lca
                    path_u = []
                    cur = u
                    while cur != lca:
                        path_u.append(cur)
                        cur = parent[cur]
                    path_u.append(lca)
                    # Path from v to lca
                    path_v = []
                    cur = v
                    while cur != lca:
                        path_v.append(cur)
                        cur = parent[cur]
                    path_v.append(lca)
                    cycle = path_u + path_v[::-1][1:]
                    if len(cycle) < g:
                        # Build edge list
                        edges = []
                        for i in range(len(cycle)):
                            a1 = cycle[i]
                            b1 = cycle[(i + 1) % len(cycle)]
                            if b1 in adj[a1]:
                                edges.append((a1, b1))
                        return cycle, edges
    return None, None


def _perform_2switch(adj, n, cycle, rng):
    """Perform a 2-switch on a cycle to break it.
    cycle: list of vertices in order.
    Returns True if switch was performed and graph remains simple."""
    L = len(cycle)
    if L < 3:
        return False
    # Pick two non-adjacent edges on the cycle
    # Edge i is (cycle[i], cycle[(i+1)%L])
    # We need edges i and j where j is not i and not (i+1)%L and not (i-1)%L
    valid_pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            # Check they are not adjacent in the cycle
            diff = (j - i) % L
            if diff == 1 or diff == L - 1:
                continue
            valid_pairs.append((i, j))
    if not valid_pairs:
        return False
    i, j = rng.choice(valid_pairs)
    a1, b1 = cycle[i], cycle[(i + 1) % L]
    a2, b2 = cycle[j], cycle[(j + 1) % L]
    # Option 1: (a1,a2) and (b1,b2)
    # Option 2: (a1,b2) and (b1,a2)
    for (new_e1, new_e2) in [((a1, a2), (b1, b2)), ((a1, b2), (b1, a2))]:
        x1, y1 = new_e1
        x2, y2 = new_e2
        if x1 == y1 or x2 == y2:
            continue
        if y1 in adj[x1] or y2 in adj[x2]:
            continue
        # Check we're not creating a multi-edge
        # Remove old edges, add new ones
        adj[a1].discard(b1)
        adj[b1].discard(a1)
        adj[a2].discard(b2)
        adj[b2].discard(a2)
        adj[x1].add(y1)
        adj[y1].add(x1)
        adj[x2].add(y2)
        adj[y2].add(x2)
        # Verify simplicity
        ok = True
        for v in range(n):
            if len(adj[v]) != 3:
                ok = False
                break
        if ok:
            return True
        # Undo
        adj[x1].discard(y1)
        adj[y1].discard(x1)
        adj[x2].discard(y2)
        adj[y2].discard(x2)
        adj[a1].add(b1)
        adj[b1].add(a1)
        adj[a2].add(b2)
        adj[b2].add(a2)
    return False


def _verify_girth(adj, n, g):
    """Verify no cycle shorter than g using BFS from each vertex."""
    for start in range(n):
        parent = [-1] * n
        dist = [-1] * n
        parent[start] = start
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Check cycle length
                    a, b = u, v
                    da, db = dist[u], dist[v]
                    while a != b:
                        if da > db:
                            a = parent[a]
                            da -= 1
                        elif db > da:
                            b = parent[b]
                            db -= 1
                        else:
                            a = parent[a]
                            b = parent[b]
                            da -= 1
                            db -= 1
                    lca = a
                    clen = dist[u] + dist[v] - 2 * dist[lca] + 1
                    if clen < g:
                        return False
    return True


def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    Returns (n, edges) where edges is a list of (u,v) tuples."""
    deadline = time.time() + 145  # 5s safety margin
    rng = random.Random(42)
    
    # Moore bound for 3-regular, girth g
    # For girth g >= 3: n >= 2 * (3*(3^(g/2-1) - 1)/(3-1)) for even g
    # For g=16: n >= 2*(3*(3^7-1)/2) = 3*(2187-1) = 6558... that's not right
    # Actually for 3-regular: n >= 2 * sum_{i=0}^{g/2-1} 3^i for even g
    # g=16: 2*(1+3+9+27+81+243+729) = 2*1093 = 2186
    # But the known cage has 936 vertices, so Moore bound is lower
    # Let me just start from reasonable sizes
    
    # Try increasing n from a reasonable lower bound
    # For girth 16, 3-regular, try n from ~200 upward
    # But we want minimum n, so try smaller first
    
    best_result = None
    best_n = float('inf')
    
    # Try different n values
    for n in range(100, 1000, 2):  # n must be even for 3-regular
        if time.time() > deadline - 10:
            break
        # Try several random restarts for this n
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            adj = _random_3_regular(n, rng)
            if adj is None:
                continue
            
            # Try to increase girth
            max_iters = 500
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                cycle, edges = _find_short_cycle(adj, n, g)
                if cycle is None:
                    # No short cycles found - verify
                    if _verify_girth(adj, n, g):
                        # Build edge list
                        edge_list = []
                        for u in range(n):
                            for v in adj[u]:
                                if u < v:
                                    edge_list.append((u, v))
                        if n < best_n:
                            best_n = n
                            best_result = (n, edge_list)
                        break
                if not _perform_2switch(adj, n, cycle, rng):
                    break  # Can't make progress
            
            if best_result is not None and best_n <= n:
                # Found something at this n or smaller, can stop trying larger
                # Actually we want minimum, so if we found at this n, larger won't help
                # But we might find smaller n later? No, we're going up.
                # Actually we should try to find the minimum, so continue searching
                # but we can break if we've found a valid one
                pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the original L-U-W construction result
    # Use the seed program's approach as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
