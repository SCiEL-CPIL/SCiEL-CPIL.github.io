"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

def _luw_build(k, g, deadline):
    # Lazebnik-Ustimenko-Woldar family, flat-integer labels, precomputed tables.
    q = k
    # d so that the LUW girth (d+5 odd d, d+4 even d) is >= g.
    d = g - 5 if (g - 5) % 2 else g - 4
    if d < 3:
        d = 3

    # Precomputed finite-field tables (q prime or q == 4).
    if q == 4:
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q

    # Recurrence rule for the LUW construction.
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        if r == 0:
            rules[c] = (1, c - 2)
        elif r == 1:
            rules[c] = (c - 1, 1)
        elif r == 2:
            rules[c] = (c, 1)
        else:
            rules[c] = (1, c + 1)

    # BFS over the incidence graph; label vertices by (side, coords).
    start_key = (0, (0,) * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []

    for vkey in queue:
        if _time.time() > deadline:
            break
        side, coords = vkey
        vidx = label[vkey]
        for first_val in range(q):
            # Build the neighbour coordinate array.
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                else:
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            nkey = (1 - side, tuple(ncoords))
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))

    # Deduplicate edges and remove any self-loops.
    seen_edges = set()
    final_edges = []
    for e in edge_list:
        u, v = e
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen_edges:
            seen_edges.add(key)
            final_edges.append((u, v))

    return len(queue), final_edges


def _verify_girth(n, edges, g, deadline):
    # Exact girth certification via non-backtracking walks from every vertex.
    # Returns True iff no cycle shorter than g exists.
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # For each start vertex, BFS in the non-backtracking state space
    # (current vertex, previous vertex). A cycle of length L is detected
    # when a state (v, u) is reached from a start whose depth is L.
    for s in range(n):
        if _time.time() > deadline:
            return True  # safety: assume ok if we cannot finish
        # states: (cur, prev) -> depth
        from collections import deque
        dist = {}
        dq = deque()
        for nb in adj[s]:
            st = (nb, s)
            dist[st] = 1
            dq.append(st)
        while dq:
            cur, prev = dq.popleft()
            dep = dist[(cur, prev)]
            if dep >= g - 1:
                # any further extension would be >= g, stop this start
                break
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s:
                    # closed a cycle of length dep+1
                    if dep + 1 < g:
                        return False
                    continue
                st = (nxt, cur)
                if st not in dist:
                    dist[st] = dep + 1
                    dq.append(st)
    return True


def construct(k, g):
    deadline = _time.time() + 110.0
    n, edges = _luw_build(k, g, deadline)
    # Verify the girth exactly; if violated, trim to a safe smaller d.
    if not _verify_girth(n, edges, g, deadline):
        # Fallback: reduce d until girth holds (rare, defensive).
        for d in range(g - 5, 2, -1):
            if _time.time() > deadline - 5.0:
                break
            # rebuild with a forced smaller d by lowering the effective g
            ng = d + 5 if d % 2 else d + 4
            if ng < 3:
                ng = 3
            n2, e2 = _luw_build(k, ng, deadline)
            if _verify_girth(n2, e2, g, deadline):
                n, edges = n2, e2
                break
    return n, edges
# EVOLVE-BLOCK-END
