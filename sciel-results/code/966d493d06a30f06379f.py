import time
import random
import sys
from collections import deque

def _moore_bound(k, g):
    if k < 2 or g < 3:
        return 0
    # Standard Moore bound
    if g % 2 == 0:
        r = g // 2 - 1
        return 2 * sum((k - 1) ** i for i in range(r + 1))
    else:
        r = (g - 1) // 2
        return k * sum((k - 1) ** i for i in range(r))

def _verify_girth(n, adj, g):
    # Exact girth verification via non-backtracking BFS from every vertex.
    # A graph has girth >= g iff there is no cycle of length < g.
    # For each start vertex, do a BFS on directed edges (u->v) tracking the
    # "previous" vertex to forbid backtracking. The first time we see a
    # directed edge (u, v) that was already reached from a different parent
    # at the same or smaller depth, we have a cycle.
    #
    # More efficient: standard BFS with parent tracking per source.
    # dist[v], parent[v]. When we find a non-tree edge u-v with v != parent[u]
    # and u != parent[v] and dist[u] + dist[v] + 1 < g, we have a short cycle.
    # But we must be careful: BFS tree from source s; an edge (u,v) that is not
    # a tree edge gives a cycle of length dist[u]+dist[v]+1 only if it's a
    # "cross" or "back" edge. For undirected simple graph, any non-tree edge
    # (u,v) with v != parent[u] and u != parent[v] yields a cycle of length
    # dist[u] + dist[v] + 1 (the two tree paths to s plus the edge).
    # We only need to detect cycles of length < g.
    #
    # We can prune: once dist[u] + 1 >= g, expanding u cannot create a cycle
    # of length < g (any cycle through u has length at least 2*dist[u]+1 >= 2g-1).
    # Actually the relevant bound: a cycle found via edge (u,v) has length
    # dist[u]+dist[v]+1. The minimum possible dist[v] (for v not parent of u)
    # is 0 (if v==s) or 1. So min cycle length from u is dist[u]+1 (if v==s)
    # or dist[u]+2. To be safe, stop expanding u when dist[u]+1 >= g.
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du + 1 >= g:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                else:
                    if dist[v] <= du:
                        # non-tree edge, not the reverse tree edge
                        # (reverse tree edge would have dist[v] == du+1 and v==parent[u], already skipped)
                        if du + dist[v] + 1 < g:
                            return False
    return True

def _verify_simple_regular(n, edges, k):
    deg = [0] * n
    seen = set()
    for (i, j) in edges:
        if i < 0 or i >= n or j < 0 or j >= n:
            return False
        if i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    return True

def _check_girth_circulant(n, steps, girth_limit):
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    queue = [0]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        du = dist[u]
        if du + 1 >= girth_limit:
            continue
        for s in steps:
            v = (u + s) % n
            if v == parent[u]:
                continue
            if dist[v] == -1:
                dist[v] = du + 1
                parent[v] = u
                queue.append(v)
            else:
                if dist[v] <= du:
                    if du + dist[v] + 1 < girth_limit:
                        return False
    return True

def _find_short_cycles(n, adj, g):
    # Return list of cycles (list of vertices) of length in [3, g-1].
    cycles = []
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du + 1 >= g:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                else:
                    if dist[v] <= du:
                        L = du + dist[v] + 1
                        if 3 <= L < g:
                            # reconstruct cycle
                            path_uv = []
                            x = u
                            while x != s:
                                path_uv.append(x)
                                x = parent[x]
                            path_uv.append(s)
                            path_uv.reverse()  # s -> ... -> u
                            path_sv = []
                            x = v
                            while x != s:
                                path_sv.append(x)
                                x = parent[x]
                            path_sv.append(s)
                            path_sv.reverse()  # s -> ... -> v
                            # cycle: u -> ... -> s -> ... -> v -> u
                            cyc = path_uv + path_sv[1:]
                            cycles.append(cyc)
    return cycles

def _try_repair(n, adj, g, deadline):
    # Iteratively find short cycles and rewire to break them.
    # adj: list of sets of neighbors.
    # We perform 2-switches: pick two edges (a,b) and (c,d) with no shared
    # endpoints and no new edges, and replace with (a,c),(b,d) or (a,d),(b,c).
    # A 2-switch preserves degrees and edge count. We only accept it if it
    # doesn't create a new shorter cycle and reduces the number of short cycles.
    for _ in range(2000):
        if time.time() > deadline:
            return False
        cycles = _find_short_cycles(n, adj, g)
        if not cycles:
            return True
        cyc = cycles[0]
        # try to break this cycle by rewiring an edge on it with an edge off it
        # Pick an edge (u,v) on the cycle. Find a non-adjacent vertex w such that
        # rewiring (u,v) with (w, z) for some z breaks the cycle.
        # Simpler: take edge (u,v) on cycle. Find a vertex w not adjacent to u or v,
        # and a vertex z adjacent to w, not on the problematic structure, such that
        # (u,w) and (v,z) are both non-edges and not creating duplicates.
        # Actually a clean 2-switch: edges (u,v) and (w,z) -> (u,w) and (v,z).
        # Need: w,z not adjacent to u or v respectively in a way that creates dup.
        # Conditions: w != u,v; z != u,v; (u,w) not edge; (v,z) not edge;
        #            (u,w) and (v,z) not already present; w != z.
        # And the switch should reduce short cycles.
        for _attempt in range(200):
            if time.time() > deadline:
                return False
            # pick an edge on the cycle
            idx = random.randrange(len(cyc))
            u = cyc[idx]
            v = cyc[(idx + 1) % len(cyc)]
            # find a random edge (w,z) not incident to u or v
            w = random.randrange(n)
            if w == u or w == v:
                continue
            # pick z adjacent to w, z not u,v, and (u,w),(v,z) non-edges
            zw = list(adj[w])
            if not zw:
                continue
            z = random.choice(zw)
            if z == u or z == v or z == w:
                continue
            if v in adj[u]:  # they are adjacent (they are, on cycle)
                pass
            if w in adj[u] or z in adj[v]:
                continue
            if w == z:
                continue
            # also ensure (u,w) and (v,z) are not already edges (checked above)
            # and (u,z),(v,w) not needed for this orientation
            # perform switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(z)
            adj[z].remove(w)
            adj[u].add(w)
            adj[w].add(u)
            adj[v].add(z)
            adj[z].add(v)
            # check it didn't create a new shorter cycle (quick local check)
            # For now, accept; the outer loop will re-check.
            break
        else:
            # couldn't find a switch this pass; try a random 2-switch anywhere
            for _attempt in range(50):
                if time.time() > deadline:
                    return False
                u = random.randrange(n)
                v = random.choice(list(adj[u]))
                w = random.randrange(n)
                if w in adj[u] or w == u or w == v:
                    continue
                zw = list(adj[w])
                if not zw:
                    continue
                z = random.choice(zw)
                if z in adj[u] or z == u or z == v or z == w:
                    continue
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(z)
                adj[z].remove(w)
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(z)
                adj[z].add(v)
                break
    return False

def _random_regular(n, k, deadline):
    # Configuration model to build a random k-regular multigraph, then
    # dedupe to a simple graph if possible. For k=5, n large, collisions are rare.
    # Build half-edges: each vertex has k stubs. Randomly pair stubs.
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    random.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            # self-loop; retry by swapping with a later stub
            for j in range(i + 2, len(stubs), 2):
                c, d = stubs[j], stubs[j + 1]
                if c != a and d != a and c != b and d != b:
                    stubs[i], stubs[j] = stubs[j], stubs[i]
                    stubs[i + 1], stubs[j + 1] = stubs[j + 1], stubs[i + 1]
                    a, b = stubs[i], stubs[i + 1]
                    break
            if a == b:
                return None
        if b in adj[a]:
            # parallel edge; try to fix by swapping
            fixed = False
            for j in range(i + 2, len(stubs), 2):
                c, d = stubs[j], stubs[j + 1]
                if c != a and d != a and c != b and d != b and c not in adj[a] and d not in adj[a] and c != d:
                    stubs[i], stubs[j] = stubs[j], stubs[i]
                    stubs[i + 1], stubs[j + 1] = stubs[j + 1], stubs[i + 1]
                    a, b = stubs[i], stubs[i + 1]
                    fixed = True
                    break
            if not fixed:
                return None
        adj[a].add(b)
        adj[b].add(a)
    # verify all degrees == k
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return adj

def _adj_to_edges(n, adj):
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges

def _construct_random_repair(k, g, n_start, deadline):
    # Try n from n_start upward; for each, build a random k-regular graph and
    # repair short cycles. Return the smallest valid (n, edges) found.
    best = None
    # For k=5, n must be even (k*n even => 5n even => n even)
    if k % 2 == 1:
        if n_start % 2 != 0:
            n_start += 1
    for n in range(n_start, n_start + 4000):
        if time.time() > deadline:
            break
        if k % 2 == 1 and n % 2 != 0:
            continue
        # a few attempts per n
        for attempt in range(4):
            if time.time() > deadline:
                break
            adj = _random_regular(n, k, deadline)
            if adj is None:
                continue
            if _try_repair(n, adj, g, deadline):
                if _verify_girth(n, adj, g):
                    edges = _adj_to_edges(n, adj)
                    if _verify_simple_regular(n, edges, k):
                        best = (n, edges)
                        return best
    return best

def _construct_luw(k, g):
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    moore_bound = _moore_bound(k, g)
    if moore_bound % 2 != 0:
        moore_bound += 1

    best = None

    # 1) Try circulants (fast, small n candidates)
    if k == 5 and g == 9:
        max_n_search = 1500
        for n in range(moore_bound, max_n_search, 2):
            if time.time() > deadline:
                break
            candidates = []
            structured_pairs = [
                (n // 5, n // 3), (n // 4, n // 3), (n // 6, n // 4),
                (n // 5, n // 4), (n // 7, n // 3), (n // 6, n // 5),
                (n // 8, n // 3), (n // 5, n // 7), (n // 4, n // 5),
                (n // 6, n // 3), (n // 7, n // 5), (n // 8, n // 5),
            ]
            for s1, s2 in structured_pairs:
                if 0 < s1 < n // 2 and 0 < s2 < n // 2 and s1 != s2 and s1 + s2 != n:
                    candidates.append((s1, s2))
            for _ in range(30):
                s1 = random.randint(1, n // 2 - 1)
                s2 = random.randint(1, n // 2 - 1)
                if s1 != s2 and s1 + s2 != n:
                    candidates.append((s1, s2))
            for s1, s2 in candidates:
                if time.time() > deadline:
                    break
                if s1 == s2 or s1 == 0 or s2 == 0 or s1 == n // 2 or s2 == n // 2:
                    continue
                if s1 + s2 == n:
                    continue
                steps = [n // 2, s1, -s1, s2, -s2]
                if _check_girth_circulant(n, steps, g):
                    edges = []
                    for i in range(n):
                        for s in steps:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    if _verify_simple_regular(n, edges, k):
                        best = (n, edges)
                        return best

    # 2) Random regular graph + repair, starting near Moore bound
    rr = _construct_random_repair(k, g, moore_bound, deadline)
    if rr is not None:
        if best is None or rr[0] < best[0]:
            best = rr

    # 3) LUW fallback
    if time.time() < deadline:
        luw = _construct_luw(k, g)
        if best is None or luw[0] < best[0]:
            best = luw

    if best is None:
        # last resort: LUW regardless of time
        best = _construct_luw(k, g)

    return best
