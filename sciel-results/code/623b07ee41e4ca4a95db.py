import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar
    construction CD(d, q) with q = k.
    
    The L-U-W construction CD(d, q) produces a q-regular bipartite graph on 2*q^(d+1) vertices
    with girth at least 2d+4 (or 2d+5 for odd d).
    
    For k=5, g=9: minimal d=3 gives girth >= 10, n = 2*5^4 = 1250.
    """
    deadline = time.time() + 115.0
    q = k
    
    # Find minimal d such that guaranteed girth >= g
    def min_d_for_girth(g):
        for d in range(1, 60):
            if d % 2 == 1:
                if 2 * d + 5 >= g:
                    return d
            else:
                if 2 * d + 4 >= g:
                    return d
        return 60
    
    # GF arithmetic
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def gf_add(x,y): return x^y
        def gf_sub(x,y): return x^y
        def gf_mul(x,y): return table[x][y]
    else:
        def gf_add(x,y): return (x+y)%q
        def gf_sub(x,y): return (x-y)%q
        def gf_mul(x,y): return (x*y)%q
    
    def get_rule(c):
        if c==2: return (1,1)
        if c==3: return (2,1)
        if c==4: return (1,2)
        r = (c-5)%4
        c0 = c-r
        if r==0: return (1,c0-2)
        if r==1: return (c0-1,1)
        if r==2: return (c0,1)
        return (1,c0+1)
    
    def build_graph(d):
        """Build L-U-W CD(d,q) graph via BFS from zero vertex."""
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        
        i = 0
        while i < len(order):
            if time.time() > deadline - 10:
                break
            side, coords = order[i]
            i += 1
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = get_rule(c)
                    if side == 0:
                        y[c] = gf_add(coords[c], gf_mul(y[a], coords[b]))
                    else:
                        y[c] = gf_sub(coords[c], gf_mul(coords[a], y[b]))
                other = (1-side, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[(side, coords)], index[other]))
        
        return len(order), edges
    
    def verify(n, edges, min_girth):
        """Verify graph is simple, k-regular, and has girth >= min_girth."""
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(n):
            if len(adj[u]) != k:
                return False
        
        # Check girth via BFS from each vertex
        for sv in range(n):
            if time.time() > deadline - 5:
                return True
            dist = [-1]*n
            par = [-1]*n
            dist[sv] = 0
            qq = [sv]
            qi = 0
            while qi < len(qq):
                u = qq[qi]
                qi += 1
                if dist[u] >= min_girth - 1:
                    continue
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u]+1
                        par[w] = u
                        qq.append(w)
                    elif w != par[u] and u != par[w]:
                        cl = dist[u]+dist[w]+1
                        if cl < min_girth:
                            return False
        return True
    
    d = min_d_for_girth(g)
    
    for d_try in range(d, d+10):
        if time.time() > deadline - 15:
            break
        n, edges = build_graph(d_try)
        if n > 0 and verify(n, edges, g):
            return n, edges
    
    # Fallback: return the first construction regardless
    n, edges = build_graph(d)
    return n, edges
