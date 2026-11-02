# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 140.0   # explicit guard; 150s kill, keep 10s safety margin

    # ---- Guaranteed-valid fallback: Lazebnik-Ustimenko-Woldar CD graph ----
    def _lw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # ---- Fast exact girth check: BFS from each vertex, detect shortest cycle ----
    def _girth(adj, n, min_check=3, max_check=None):
        """Return the girth (shortest cycle length). If no cycle < max_check, return max_check."""
        if max_check is None:
            max_check = 2 * g + 1
        # For each start vertex, do BFS tracking parent to find shortest cycle
        # A cycle of length L exists iff during BFS from s, we find an edge (u,v)
        # where dist[u] + 1 + dist[v] = L and u != parent[v] (non-backtracking)
        # More precisely: BFS from s, for each edge (u,v) not on the tree path,
        # cycle length = dist[u] + 1 + dist[v] if u and v are at different depths
        # or 2*dist[u]+1 if u==v (odd cycle through s)
        # Standard approach: for each s, BFS; for each non-tree edge (u,v), 
        # if u != parent[v] and v != parent[u], cycle = dist[u]+1+dist[v]
        best = max_check
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            dq = deque([s])
            while dq:
                u = dq.popleft()
                if dist[u] + 1 >= best:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        dq.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # non-tree edge: cycle length = dist[u] + 1 + dist[v]
                        cyc = dist[u] + 1 + dist[v]
                        if cyc >= 3 and cyc < best:
                            best = cyc
            if best <= min_check:
                break
        return best

    # ---- Moore-bound girth-10 tree, leaves identified -> 4-regular graph ----
    def _build_tree(k, g):
        depth = (g - 2) // 2
        adj = {}
        def new_node():
            idx = len(adj)
            adj[idx] = []
            return idx
        def add_edge(a, b):
            adj[a].append(b)
            adj[b].append(a)
        root = new_node()
        frontier = [root]
        for lvl in range(depth):
            nxt = []
            for v in frontier:
                for _ in range(k):
                    c = new_node()
                    add_edge(v, c)
                    nxt.append(c)
            frontier = nxt
        leaves = frontier
        return adj, root, leaves

    def _identify(adj, leaves, perm):
        n_int = len(adj)
        L = len(leaves)
        new_id = list(range(n_int))
        for i in range(L):
            p = perm[i]
            pair = min(i, p)
            new_id[leaves[i]] = n_int + pair
        new_n = n_int + L // 2
        new_adj = [set() for _ in range(new_n)]
        for a in range(n_int):
            for b in adj[a]:
                na, nb = new_id[a], new_id[b]
                if na != nb:
                    new_adj[na].add(nb)
                    new_adj[nb].add(na)
        return [sorted(s) for s in new_adj]

    # ---- Search: local search over leaf permutations with fast girth scoring ----
    def _search(k, g, time_budget):
        adj, root, leaves = _build_tree(k, g)
        L = len(leaves)
        # initial: pair leaf i with leaf (i + L//2) % L
        perm = [ (i + L//2) % L for i in range(L) ]
        for i in range(L//2):
            j = (i + L//2) % L
            perm[i] = j
            perm[j] = i
        
        def build(perm):
            return _identify(adj, leaves, perm)
        
        def score(perm):
            adj2 = build(perm)
            gr = _girth(adj2, len(adj2), min_check=3, max_check=g)
            return (g - gr) if gr < g else 0, adj2, gr
        
        cur_perm = perm[:]
        cur_pen, cur_adj, cur_gr = score(cur_perm)
        best_pen, best_adj, best_gr = cur_pen, cur_adj, cur_gr
        rng = random.Random(12345)
        t_end = time.time() + time_budget
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < t_end and no_improve < max_no_improve:
            # pick two random pairs to swap
            i = rng.randrange(L//2)
            j = rng.randrange(L//2)
            if i == j:
                continue
            pi, pj = cur_perm[i], cur_perm[j]
            if pi == pj:
                continue
            new_perm = cur_perm[:]
            new_perm[i] = j
            new_perm[j] = i
            new_perm[pi] = pj
            new_perm[pj] = pi
            npen, nadj, ngr = score(new_perm)
            
            if npen <= cur_pen:
                cur_perm = new_perm
                cur_pen, cur_adj, cur_gr = npen, nadj, ngr
                no_improve = 0
                if npen < best_pen:
                    best_pen, best_adj, best_gr = npen, nadj, ngr
            else:
                no_improve += 1
                # accept worse with small probability (annealing-like)
                if rng.random() < 0.05 and npen < cur_pen + 3:
                    cur_perm = new_perm
                    cur_pen, cur_adj, cur_gr = npen, nadj, ngr
        
        return best_adj

    # ---- Driver ----
    try:
        time_budget = min(110.0, max(2.0, (deadline - time.time()) - 25.0))
        adj = _search(k, g, time_budget)
        n = len(adj)
        ok = True
        for v in range(n):
            if len(adj[v]) != k:
                ok = False
                break
        if ok:
            gr = _girth(adj, n, min_check=3, max_check=g)
            if gr >= g:
                edges = []
                for v in range(n):
                    for w in adj[v]:
                        if v < w:
                            edges.append((v, w))
                return n, edges
    except Exception:
        pass
    try:
        return _lw_fallback()
    except Exception:
        return _lw_fallback()
# EVOLVE-BLOCK-END
