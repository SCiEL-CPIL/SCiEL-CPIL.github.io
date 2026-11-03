"""Construct a k-regular graph of girth >= g with as few vertices as possible."""
import time
import random
from collections import deque

_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)


def _girth_ok(adj, g):
    """Return True iff graph (adjacency list) has no cycle of length < g.
    Exact: BFS from each vertex to depth h=(g-1)//2; check edges closing a
    cycle of length <= 2h (i.e. < g)."""
    n = len(adj)
    h = (g - 1) // 2
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= h:
                continue
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = du + 1
                    parent[w] = u
                    if du + 1 < h:
                        q.append(w)
        # check edges (u,w) with dist(u)+dist(w)+1 < g and u,w in different branches
        for u in range(n):
            du = dist[u]
            if du < 0 or du > h:
                continue
            for w in adj[u]:
                dw = dist[w]
                if dw < 0 or dw > h:
                    continue
                if du + dw + 1 < g:
                    # ensure different branches (no ancestor relation)
                    # walk up from u and w up to min depth
                    a, b = u, w
                    # find LCA-ish: if one is ancestor of other, skip
                    # collect ancestors of the shallower
                    if du <= dw:
                        x, y = u, w
                        depth = du
                    else:
                        x, y = w, u
                        depth = dw
                    anc = set()
                    cur = x
                    for _ in range(depth):
                        anc.add(cur)
                        cur = parent[cur]
                    cur = y
                    for _ in range(depth):
                        if cur in anc:
                            break
                        cur = parent[cur]
                    else:
                        # no common ancestor within depth -> different branches
                        return False
    return True


def _luw_component(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d,q), q=k, smallest component."""
    q = k
    d = g - 4 if (g - 5) % 2 else g - 5
    if d < 3:
        d = 3
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x ^ y
        sub = lambda x,y: x ^ y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x + y) % q
        sub = lambda x,y: (x - y) % q
        mul = lambda x,y: x * y % q
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5) % 4, c - (c-5) % 4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield (1-side), tuple(y)

    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _random_regular(n, k, rng):
    """Configuration model -> simple k-regular graph (adjacency list) or None."""
    if n * k % 2:
        return None
    half = list(range(n)) * (k // 2) if k % 2 == 0 else None
    # general: build stub list
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    # random matching
    for attempt in range(50):
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i+1]
            if a == b:
                ok = False
                break
            if b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if ok:
            return [sorted(a) for a in adj]
    return None


def _two_switch(adj, i, j, a, b, rng):
    """Replace edges (i,a),(j,b) with (i,b),(j,a) if valid and degree-preserving."""
    if b in adj[i] or a in adj[j]:
        return False
    adj[i].remove(a); adj[j].remove(b)
    adj[i].add(b); adj[j].add(a)
    return True


def _repair_girth(adj, g, max_iter, rng):
    """Try to raise girth via 2-switches. Returns new adj."""
    n = len(adj)
    for _ in range(max_iter):
        if _time_left() < 3:
            break
        if _girth_ok(adj, g):
            break
        # find a short cycle, pick an edge on it, try a 2-switch breaking it
        # locate a cycle < g via BFS from a random vertex
        s = rng.randrange(n)
        h = (g-1)//2
        dist = [-1]*n; parent=[-1]*n
        dist[s]=0; q=deque([s])
        cyc_edge = None
        while q:
            u=q.popleft()
            if dist[u]>=h: continue
            for w in adj[u]:
                if dist[w]==-1:
                    dist[w]=dist[u]+1; parent[w]=u
                    if dist[u]+1<h: q.append(w)
        for u in range(n):
            if dist[u]<0 or dist[u]>h: continue
            for w in adj[u]:
                if dist[w]<0 or dist[w]>h: continue
                if dist[u]+dist[w]+1 < g:
                    # different branches?
                    if dist[u]<=dist[w]:
                        x,y,d0 = u,w,dist[u]
                    else:
                        x,y,d0 = w,u,dist[w]
                    anc=set(); cur=x
                    for _ in range(d0):
                        anc.add(cur); cur=parent[cur]
                    cur=y; found=False
                    for _ in range(d0):
                        if cur in anc: found=True; break
                        cur=parent[cur]
                    if not found:
                        cyc_edge=(u,w); break
            if cyc_edge: break
        if not cyc_edge:
            break
        u,w = cyc_edge
        # try to 2-switch using u with a neighbour of some other vertex
        # pick another edge (j, b) where j not u, b not in adj[u]
        done=False
        for _t in range(20):
            j = rng.randrange(n)
            if j==u: continue
            b = rng.choice(adj[j])
            if b==w or b in adj[u]: continue
            a = w
            # remove (u,a),(j,b) add (u,b),(j,a)
            if a in adj[u] and b in adj[j]:
                adj[u].remove(a); adj[j].remove(b)
                adj[u].add(b); adj[j].add(a)
                done=True; break
        if not done:
            break
    return adj


def construct(k, g):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    # Moore bound (even)
    if g % 2 == 0:
        # even girth: 2 * sum_{i=0}^{g/2-1} (k-1)^i
        s = 0
        for i in range(g//2):
            s += (k-1)**i
        moore = 2*s
    else:
        s = 1
        for i in range(1, (g+1)//2):
            s += (k-1)**i
        moore = s
    # ensure even for regular
    if moore % 2:
        moore += 1

    best = None
    # Candidate A: LUW (guaranteed)
    try:
        n_luw, e_luw = _luw_component(k, g)
        best = (n_luw, e_luw)
    except Exception:
        pass

    # Candidate B: randomized search for smaller n
    n = moore
    budget_n = max(moore, (best[0] if best else 10**9) - 2)
    step = 2
    while n <= budget_n and _time_left() > 15:
        # a few random restarts at this n
        for restart in range(6):
            if _time_left() < 10:
                break
            adj = _random_regular(n, k, rng)
            if adj is None:
                continue
            adj = _repair_girth(adj, g, 200, rng)
            if _girth_ok(adj, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                cand = (n, edges)
                if best is None or n < best[0]:
                    best = cand
                break  # found at this n, stop increasing
        if best is not None and best[0] == n:
            break
        n += step

    if best is None:
        # fallback: LUW again
        n_luw, e_luw = _luw_component(k, g)
        best = (n_luw, e_luw)

    return best[0], best[1]
