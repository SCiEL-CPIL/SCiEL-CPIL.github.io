import time
from collections import deque

def _has_short_cycle(edges, n, g):
    """Return True if the graph has a cycle of length < g.
    Uses BFS from each vertex up to depth (g-2)//2, detecting non-tree edges."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    max_depth = (g - 2) // 2
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                else:
                    # Non-tree edge (u,v): cycle length = du + dist[v] + 1
                    if v != parent[u] and u != parent[v]:
                        if du + dist[v] + 1 < g:
                            return True
    return False

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction. For k=4, g=10 gives 320 vertices."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _circulant_edges(n, S):
    """Build edge list for circulant C(n; S) where S is a set of symmetric jumps."""
    edges = []
    edge_set = set()
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if i < j:
                key = (i, j)
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append(key)
    return edges

def _has_4cycle_circulant(n, S):
    """Quick check: does C(n; S) have a 4-cycle?
    A 4-cycle exists iff there are two distinct vertices with >= 2 common neighbors.
    For circulant, vertex 0 and vertex d have common neighbors iff
    the sets {s mod n : s in S} and {(d+s) mod n : s in S} intersect in >= 2 elements.
    Equivalently, there exist s1,s2,s3,s4 in S (with s1+s4 = s2+s3 mod n, s1!=s2) giving a 4-cycle.
    Simpler: check if any pair of elements in S has a difference that is also expressible as a difference of two elements in S (with distinct pairs)."""
    Sset = set()
    for s in S:
        Sset.add(s % n)
        Sset.add((-s) % n)
    # For each pair (a,b) in Sset, a-b mod n is a "chord". A 4-cycle exists if
    # there are two distinct pairs giving the same chord, i.e., a-b = c-d mod n with {a,b} != {c,d}.
    chords = {}
    elems = sorted(Sset)
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            diff = (elems[i] - elems[j]) % n
            if diff not in chords:
                chords[diff] = []
            chords[diff].append((i, j))
    for diff, pairs in chords.items():
        if len(pairs) >= 2:
            return True
    return False

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k == 4 and g == 10:
        # Search for a valid 4-regular graph of girth 10 with n < 320.
        # Try circulant graphs C(n; {±1, ±a}) and C(n; {±a, ±b}).
        best_n, best_edges = None, None

        for n in range(241, 320):
            if time.time() > deadline - 10:
                break

            # Type 1: {±1, ±a}
            for a in range(2, n // 2 + 1):
                if time.time() > deadline - 10:
                    break
                S = {1, -1, a, -a}
                if len(S) != 4:
                    continue
                # Quick 4-cycle check
                if _has_4cycle_circulant(n, S):
                    continue
                edges = _circulant_edges(n, S)
                if len(edges) != 2 * n:
                    continue
                if not _has_short_cycle(edges, n, g):
                    return n, edges

            # Type 2: {±a, ±b} with 2 <= a < b < n/2
            for a in range(2, n // 3 + 1):
                if time.time() > deadline - 10:
                    break
                for b in range(a + 1, n // 2 + 1):
                    if time.time() > deadline - 10:
                        break
                    S = {a, -a, b, -b}
                    if len(S) != 4:
                        continue
                    if _has_4cycle_circulant(n, S):
                        continue
                    edges = _circulant_edges(n, S)
                    if len(edges) != 2 * n:
                        continue
                    if not _has_short_cycle(edges, n, g):
                        return n, edges

        # Fallback: LUW construction (320 vertices)
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)
