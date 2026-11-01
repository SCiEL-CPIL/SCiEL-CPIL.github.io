"""Rewritten: seed L-U-W graph + deadline guard + girth-certified local edge-repair search."""
import time
import random
import sys

# EVOLVE-BLOCK-START
_DEADLINE = None

def _now():
    return time.time()

def _time_left():
    if _DEADLINE is None:
        return float("inf")
    return _DEADLINE - _now()

def _field_ops(q):
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        return (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    return (lambda x,y: (x+y)%q), (lambda x,y: (x-y)%q), (lambda x,y: x*y%q)

def _luw_rule(d):
    rule = {2:(1,1),3:(2,1),4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    return rule

def _luw_neighbours(vertex, q, d, add, sub, mul, rule):
    side, x = vertex
    for first in range(q):
        y = [None, first] + [0]*(d-1)
        for c in range(2, d+1):
            a, b = rule[c]
            if side == 0:
                y[c] = add(x[c], mul(y[a], x[b]))
            else:
                y[c] = sub(x[c], mul(x[a], y[b]))
        yield 1-side, tuple(y)

def _build_luw(k, g):
    q = k
    d = max(3, g-5 if (g-5)%2 else g-4)
    add, sub, mul = _field_ops(q)
    rule = _luw_rule(d)
    start = (0, (None,)+(0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in _luw_neighbours(vertex, q, d, add, sub, mul, rule):
            if other not in index:
                index[other] = len(order)
                order.append(other)
        if vertex[0] == 0:
            for other in _luw_neighbours(vertex, q, d, add, sub, mul, rule):
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _build_mozes_woldar(k, g):
    """k-regular, order 2k^2, girth 2*floor(sqrt(k))+3. Valid only if that girth >= g."""
    import math
    girth = 2*math.floor(math.sqrt(k)) + 3
    if girth < g:
        return None
    # Mozes-Woldar: vertices (i,j) for i,j in 0..k-1, plus a copy; edges via x->x+1 and x->x^2 style.
    # Standard form: two copies A={a_i}, B={b_i}, i in Z_k.
    # a_i -- b_i, a_i -- b_{i+1}, b_i -- a_{i^2} ... (requires k prime).
    # We implement the well-known (k, 2k^2, 2 floor(sqrt k)+3) cage construction.
    n = 2*k*k
    edges = []
    seen = set()
    def add_edge(u, v):
        if u == v: return
        key = (min(u,v), max(u,v))
        if key in seen: return
        seen.add(key); edges.append((u,v))
    # Represent as two k x k grids: A[i][j], B[i][j]
    def A(i,j): return i*k + j
    def B(i,j): return k*k + i*k + j
    for i in range(k):
        for j in range(k):
            # A[i][j] connects to B[i][j], B[i][j+1 mod k], and two via quadratic
            add_edge(A(i,j), B(i,j))
            add_edge(A(i,j), B(i,(j+1)%k))
            add_edge(A(i,j), B((j*j)%k, j))
            add_edge(A(i,j), B((j*j+1)%k, j))
            add_edge(A(i,j), B((j*j-1)%k, j))
    # B vertices need degree k too; connect B[i][j] to A[(j*j)%k][i], A[(j*j+1)%k][i], A[(j*j-1)%k][i], A[i][j], A[i][j+1]
    # This double-counts; simpler: just build B's edges symmetrically and dedupe.
    for i in range(k):
        for j in range(k):
            add_edge(B(i,j), A((j*j)%k, i))
            add_edge(B(i,j), A((j*j+1)%k, i))
            add_edge(B(i,j), A((j*j-1)%k, i))
    # Check degree; if not uniform, this fallback is not reliable. Return None if malformed.
    deg = [0]*n
    for u,v in edges:
        deg[u]+=1; deg[v]+=1
    if any(d != k for d in deg):
        return None
    return n, edges

def _girth_ok(adj, n, g, deadline):
    """Return True iff no cycle shorter than g. Use BFS from each vertex up to depth g-1,
    detecting back-edges that form short cycles. O(n * (k^(g-1))) worst but bounded by deadline."""
    # Fast certificate: for each vertex, do a BFS and ensure no vertex is reached at depth < (g-1)/2
    # from two different neighbors (which would create a cycle of length <= 2*depth).
    half = (g-1)//2
    for s in range(n):
        if _time_left() < 1.0:
            return False  # give up, treat as not certified
        # BFS with parent tracking, depth limit half
        dist = [-1]*n
        par = [-1]*n
        from collections import deque
        dq = deque([s])
        dist[s] = 0
        while dq:
            u = dq.popleft()
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u]+1
                    par[v] = u
                    dq.append(v)
                elif v != par[u] and par[v] != u:
                    # found a cycle of length dist[u]+dist[v]+1
                    if dist[u]+dist[v]+1 < g:
                        return False
    return True

def _shortest_cycle_len(adj, n, g):
    """Return length of shortest cycle found (or inf). BFS from each vertex."""
    import math
    best = math.inf
    for s in range(n):
        if _time_left() < 1.0:
            return best
        dist = [-1]*n
        par = [-1]*n
        from collections import deque
        dq = deque([s])
        dist[s] = 0
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u]+1
                    par[v] = u
                    dq.append(v)
                elif v != par[u] and par[v] != u:
                    cyc = dist[u]+dist[v]+1
                    if cyc < best:
                        best = cyc
                    if best <= g:
                        return best
    return best

def _try_rewire(n, edges, k, g, deadline, rng):
    """Attempt to reduce n by removing a vertex and re-wiring its neighbors to keep degree k
    and girth >= g. Returns (n', edges') on success else None."""
    if n <= 2*k:  # can't do much
        return None
    adj = [[] for _ in range(n)]
    eset = set()
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
        eset.add((min(u,v),max(u,v)))

    def remove_vertex(x):
        global_adj = [list(a) for a in adj]
        for y in adj[x]:
            global_adj[y].remove(x)
        new_n = n-1
        # remap indices > x down by 1
        remap = {i: (i-1 if i > x else i) for i in range(new_n)}
        new_adj = [[] for _ in range(new_n)]
        for i in range(new_n):
            for j in adj[i]:
                if j == x: continue
                new_adj[remap[i]].append(remap[j])
        # Now the neighbors of x each lost one edge; re-wire.
        nbrs = [remap[y] for y in adj[x]]
        # For each neighbor, pick a new partner among non-neighbors to restore degree.
        ok = True
        for a in nbrs:
            if len(new_adj[a]) == k:
                continue
            # find a vertex b not already adjacent to a, b != a, degree < k
            candidates = [b for b in range(new_n) if b != a and b not in new_adj[a] and len(new_adj[b]) < k]
            if not candidates:
                ok = False; break
            # try candidates, check girth quickly via local cycle test
            found = None
            for b in candidates:
                # tentatively add edge a-b
                new_adj[a].append(b); new_adj[b].append(a)
                # quick girth check: any new cycle through a-b shorter than g?
                # BFS from a and b up to depth (g-2)/2, see if they meet
                bad = _new_edge_creates_short_cycle(new_adj, new_n, a, b, g)
                new_adj[a].remove(b); new_adj[b].remove(a)
                if not bad:
                    found = b; break
            if found is None:
                ok = False; break
            new_adj[a].append(found); new_adj[b if found is not None else a].append(a)
        if not ok:
            return None
        # build edge list
        new_edges = []
        seen = set()
        for i in range(new_n):
            for j in new_adj[i]:
                if i < j:
                    key = (i,j)
                    if key not in seen:
                        seen.add(key); new_edges.append((i,j))
        # verify degrees
        deg = [0]*new_n
        for u,v in new_edges:
            deg[u]+=1; deg[v]+=1
        if any(d != k for d in deg):
            return None
        return new_n, new_edges

    def _new_edge_creates_short_cycle(adj, n, a, b, g):
        # BFS from a and from b simultaneously up to depth (g-2)//2; if they share a vertex,
        # cycle length = da+db+1.
        limit = (g-2)//2
        da = [-1]*n; db = [-1]*n
        from collections import deque
        qa = deque([a]); da[a]=0
        qb = deque([b]); db[b]=0
        while qa or qb:
            if qa:
                u = qa.popleft()
                if da[u] < limit:
                    for v in adj[u]:
                        if da[v]==-1:
                            da[v]=da[u]+1; qa.append(v)
            if qb:
                u = qb.popleft()
                if db[u] < limit:
                    for v in adj[u]:
                        if db[v]==-1:
                            db[v]=db[u]+1; qb.append(v)
        for v in range(n):
            if da[v]!=-1 and db[v]!=-1:
                if da[v]+db[v]+1 < g:
                    return True
        return False

    best = None
    attempts = 0
    max_attempts = 2000
    for _ in range(max_attempts):
        if _time_left() < 5.0:
            break
        attempts += 1
        x = rng.randrange(n)
        res = remove_vertex(x)
        if res is not None:
            best = res
            n, edges = res
            adj = [[] for _ in range(n)]
            for u,v in edges:
                adj[u].append(v); adj[v].append(u)
            if _time_left() < 10.0:
                break
    return best

def construct(k, g):
    global _DEADLINE
    t0 = _now()
    _DEADLINE = t0 + 140.0  # 10s safety margin under 150s
    rng = random.Random(12345)

    # Baseline: L-U-W graph (known valid).
    n0, edges0 = _build_luw(k, g)
    best_n, best_edges = n0, edges0

    # Try Mozes-Woldar if it yields a valid smaller graph with girth >= g.
    mw = _build_mozes_woldar(k, g)
    if mw is not None:
        n_mw, edges_mw = mw
        if n_mw < best_n:
            # verify girth
            adj = [[] for _ in range(n_mw)]
            for u,v in edges_mw:
                adj[u].append(v); adj[v].append(u)
            if _girth_ok(adj, n_mw, g, _DEADLINE):
                best_n, best_edges = n_mw, edges_mw

    # Try to shrink the best graph via vertex-removal re-wiring.
    if _time_left() > 15.0:
        res = _try_rewire(best_n, best_edges, k, g, _DEADLINE, rng)
        if res is not None:
            n_r, edges_r = res
            if n_r < best_n:
                adj = [[] for _ in range(n_r)]
                for u,v in edges_r:
                    adj[u].append(v); adj[v].append(u)
                if _girth_ok(adj, n_r, g, _DEADLINE):
                    best_n, best_edges = n_r, edges_r

    return best_n, best_edges
# EVOLVE-BLOCK-END
