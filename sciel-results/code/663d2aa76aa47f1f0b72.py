# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    deadline = time.time() + 140.0

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

    def _fast_girth_check(adj, n, g_target):
        """Check if girth >= g_target using non-backtracking walks.
        Returns True if no cycle shorter than g_target found."""
        if n < 3:
            return True
        # For each start vertex, do non-backtracking BFS up to depth (g_target-1)
        # A cycle of length L < g_target exists iff some vertex is reached at two
        # different non-backtracking walk lengths from the same start, where the
        # shorter length + the edge back creates a cycle < g_target.
        # Actually: a non-backtracking walk of length L from s to v where v was
        # already seen at depth d < L means there's a cycle of length L-d+1 (if we
        # can close it). More precisely, if (prev, v) was seen at depth d, and now
        # we're at depth L with (prev2, v), the cycle length is L - d + 1 only if
        # the paths don't share edges. The standard approach: track first visit depth
        # for each (prev, cur) state. If we revisit (cur, nb) at a later depth, the
        # cycle length is current_depth - first_depth + 1.
        
        from collections import deque
        max_depth = g_target - 1  # we only care about cycles of length < g_target
        
        for s in range(n):
            # state: (prev, cur), dist = walk length
            first_seen = {}  # (prev, cur) -> first depth reached
            q = deque()
            for nb in adj[s]:
                key = (s, nb)
                first_seen[key] = 1
                q.append((s, nb, 1))
            
            while q:
                prev, cur, d = q.popleft()
                if d >= max_depth:
                    continue  # don't extend further; cycles of length >= g_target are OK
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    key = (cur, nb)
                    if key in first_seen:
                        # Revisited: cycle length = d + 1 - first_seen[key]
                        cyc = d + 1 - first_seen[key]
                        if cyc < g_target and cyc >= 3:
                            return False
                    else:
                        first_seen[key] = d + 1
                        q.append((cur, nb, d + 1))
        return True

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
        leaf_pos = {}
        for i in range(L):
            leaf_pos[leaves[i]] = i
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

    def _quick_girth_penalty(adj, n, g_target):
        """Fast check: does this graph have any cycle shorter than g_target?
        Returns (penalty, actual_girth_estimate). Lower penalty is better."""
        if not _fast_girth_check(adj, n, g_target):
            # Find the actual short cycle length for better gradient
            # Use a simpler BFS-based girth estimate
            from collections import deque
            best_girth = g_target
            for s in range(n):
                dist = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    v = q.popleft()
                    if dist[v] >= g_target - 2:
                        continue
                    for nb in adj[v]:
                        if dist[nb] == -1:
                            dist[nb] = dist[v] + 1
                            q.append(nb)
                        elif dist[nb] <= dist[v]:
                            cyc = dist[v] + dist[nb] + 1
                            if cyc < best_girth:
                                best_girth = cyc
            return (g_target - best_girth), best_girth
        return 0, g_target

    def _search(k, g, time_budget):
        adj, root, leaves = _build_tree(k, g)
        L = len(leaves)
        perm = [0] * L
        for i in range(L // 2):
            j = (i + L // 2) % L
            perm[i] = j
            perm[j] = i
        
        def build(perm):
            return _identify(adj, leaves, perm)
        
        cur_perm = perm[:]
        cur_adj = build(cur_perm)
        n = len(cur_adj)
        cur_pen, cur_gr = _quick_girth_penalty(cur_adj, n, g)
        best_pen, best_adj, best_gr = cur_pen, cur_adj[:], cur_gr
        
        rng = random.Random(42)
        t_end = time.time() + time_budget
        cur_temp = 1.0
        
        while time.time() < t_end:
            # Random 2-swap move
            i = rng.randrange(L // 2)
            j = rng.randrange(L // 2)
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
            
            new_adj = build(new_perm)
            npen, ngr = _quick_girth_penalty(new_adj, n, g)
            
            # Accept if better or with probability based on temperature
            if npen < cur_pen:
                cur_perm = new_perm
                cur_adj = new_adj
                cur_pen, cur_gr = npen, ngr
                if npen < best_pen:
                    best_pen, best_adj, best_gr = npen, new_adj[:], ngr
            elif npen == cur_pen and rng.random() < 0.3:
                cur_perm = new_perm
                cur_adj = new_adj
                cur_pen, cur_gr = npen, ngr
            elif npen > cur_pen and npen <= cur_pen + 1:
                # Accept worse moves with decreasing probability
                accept_prob = 0.1 / (1.0 + ngr * 0.1 + cur_temp)
                if rng.random() < accept_prob:
                    cur_perm = new_perm
                    cur_adj = new_adj
                    cur_pen, cur_gr = npen, ngr
            
            cur_temp *= 0.9995
            if time.time() > t_end - 3:
                break
        
        return best_adj

    # Driver
    try:
        time_budget = min(110.0, max(5.0, (deadline - time.time()) - 20.0))
        adj = _search(k, g, time_budget)
        n = len(adj)
        ok = True
        for v in range(n):
            if len(adj[v]) != k:
                ok = False
                break
        if ok:
            if _fast_girth_check(adj, n, g):
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
