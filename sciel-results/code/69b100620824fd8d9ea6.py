"""2-switch shrink search on top of the Lazebnik-Ustimenko-Woldar seed graph."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 150 s kill
    random.seed(12345)

    # ---- Build the L-U-W seed graph (4-regular, girth >= g) ----
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
    index, order, seed_edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                seed_edges.append((index[vertex], index[other]))

    if time.time() > deadline:
        return len(order), seed_edges

    # ---- State as adjacency lists ----
    n = len(order)
    adj = [[] for _ in range(n)]
    for u, v in seed_edges:
        adj[u].append(v)
        adj[v].append(u)
    for v in range(n):
        adj[v].sort()

    def girth_of(adj, n, g):
        """Return True iff no cycle shorter than g using BFS from each vertex."""
        for s in range(n):
            dist = [-1] * n
            prev = [-1] * n
            dist[s] = 0
            queue = [s]
            found = False
            head = 0
            while head < len(queue) and not found:
                u = queue[head]
                head += 1
                du = dist[u]
                pu = prev[u]
                for w in adj[u]:
                    if w == pu:
                        continue
                    if dist[w] >= 0:
                        if du + 1 + dist[w] < g:
                            found = True
                            break
                    else:
                        dist[w] = du + 1
                        prev[w] = u
                        queue.append(w)
            if found:
                return False
        return True

    def fast_girth_check(adj, n, g):
        """Faster girth check: BFS from a single vertex to find shortest cycle."""
        # For a k-regular graph, if BFS from any vertex finds no cycle < g, 
        # we need to be more careful. Use a heuristic: check from multiple vertices.
        # Actually, the correct approach: the girth is the minimum over all vertices 
        # of the shortest cycle through that vertex. So we need to check all.
        # But we can early-terminate if we find a short cycle.
        for s in range(n):
            dist = [-1] * n
            prev = [-1] * n
            dist[s] = 0
            queue = [s]
            found = False
            head = 0
            while head < len(queue) and not found:
                u = queue[head]
                head += 1
                du = dist[u]
                pu = prev[u]
                for w in adj[u]:
                    if w == pu:
                        continue
                    if dist[w] >= 0:
                        if du + 1 + dist[w] < g:
                            found = True
                            break
                    else:
                        dist[w] = du + 1
                        prev[w] = u
                        queue.append(w)
            if found:
                return False
        return True

    # ---- Verify the seed is valid (it should be) ----
    if not girth_of(adj, n, g):
        return n, seed_edges

    best_n = n
    best_adj = [list(x) for x in adj]

    def build_edge_set(adj, n):
        es = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    es.add((u, v))
        return es

    def try_shrink_once(adj, n, g, deadline):
        """Try to remove one degree-2 vertex via a 2-switch."""
        deg2 = [v for v in range(n) if len(adj[v]) == 2]
        if not deg2:
            return None
        random.shuffle(deg2)
        edge_set = build_edge_set(adj, n)
        for v in deg2:
            a, b = adj[v]
            blocked = {v, a, b}
            candidates = []
            for (c, dd) in edge_set:
                if c in blocked or dd in blocked:
                    continue
                candidates.append((c, dd))
            if not candidates:
                continue
            random.shuffle(candidates)
            for (c, dd) in candidates:
                for (a1, b1, c1, d1) in [(a, b, c, dd), (a, b, dd, c)]:
                    if (min(a1, c1), max(a1, c1)) in edge_set:
                        continue
                    if (min(b1, d1), max(b1, d1)) in edge_set:
                        continue
                    new_adj = [list(x) for x in adj]
                    new_adj[v].remove(a); new_adj[a].remove(v)
                    new_adj[v].remove(b); new_adj[b].remove(v)
                    new_adj[c].remove(dd); new_adj[dd].remove(c)
                    new_adj[a1].append(c1); new_adj[c1].append(a1)
                    new_adj[b1].append(d1); new_adj[d1].append(b1)
                    new_adj.pop(v)
                    m = n - 1
                    remap = {x: (x if x < v else x - 1) for x in range(n) if x != v}
                    fixed = []
                    for u in range(m):
                        orig = u + (1 if u >= v else 0)
                        fixed.append([remap[w] for w in new_adj[orig]])
                    ok = True
                    seen = set()
                    for u in range(m):
                        if len(fixed[u]) != k:
                            ok = False; break
                        for w in fixed[u]:
                            if u == w:
                                ok = False; break
                            key = (min(u, w), max(u, w))
                            if key in seen:
                                ok = False; break
                            seen.add(key)
                        if not ok:
                            break
                    if not ok:
                        continue
                    if girth_of(fixed, m, g):
                        return fixed, m
        return None

    # Main loop: repeatedly shrink until stuck or deadline.
    cur_adj = [list(x) for x in adj]
    cur_n = n
    stuck_count = 0
    total_escapes = 0
    
    while time.time() < deadline:
        if cur_n < best_n:
            best_n = cur_n
            best_adj = [list(x) for x in cur_adj]
            stuck_count = 0
        res = try_shrink_once(cur_adj, cur_n, g, deadline)
        if res is None:
            stuck_count += 1
            if stuck_count > 3:
                # Try to escape with degree-preserving 2-switches
                escaped = False
                max_escapes = 50
                for _ in range(max_escapes):
                    if time.time() > deadline:
                        break
                    edges = list(build_edge_set(cur_adj, cur_n))
                    if len(edges) < 2:
                        break
                    (u1, v1), (u2, v2) = random.sample(edges, 2)
                    if len({u1, v1, u2, v2}) < 4:
                        continue
                    es_now = build_edge_set(cur_adj, cur_n)
                    if (min(u1, u2), max(u1, u2)) in es_now:
                        continue
                    if (min(v1, v2), max(v1, v2)) in es_now:
                        continue
                    cand = [list(x) for x in cur_adj]
                    cand[u1].remove(v1); cand[v1].remove(u1)
                    cand[u2].remove(v2); cand[v2].remove(u2)
                    cand[u1].append(u2); cand[u2].append(u1)
                    cand[v1].append(v2); cand[v2].append(v1)
                    if girth_of(cand, cur_n, g):
                        cur_adj = cand
                        escaped = True
                        total_escapes += 1
                        break
                if not escaped:
                    break
        else:
            cur_adj, cur_n = res

    out_edges = []
    for u in range(best_n):
        for w in best_adj[u]:
            if u < w:
                out_edges.append((u, w))
    return best_n, out_edges
# EVOLVE-BLOCK-END
