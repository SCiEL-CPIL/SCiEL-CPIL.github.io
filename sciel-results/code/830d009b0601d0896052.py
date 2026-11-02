"""De Bruijn-type finite-state construction for regular graphs of large girth."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 120.0  # seconds; evaluator kills at 150s, keep margin

    # We want a k-regular graph with girth >= g.
    # Strategy: build a directed de Bruijn graph B(k, m) where m is chosen so that
    # the girth of the resulting undirected graph (after quotienting) is >= g.
    #
    # The de Bruijn graph B(k, m) has k^m vertices (words of length m over {0..k-1}).
    # Each vertex w = (w_0, w_1, ..., w_{m-1}) has out-neighbors (w_1, ..., w_{m-1}, a) for a in {0..k-1}.
    # This is a k-regular directed graph.
    #
    # To get an undirected k-regular graph, we can use the following approach:
    # Consider the bipartite graph where left side = words of length m, right side = words of length m,
    # and connect left word (a_0,...,a_{m-1}) to right word (b_0,...,b_{m-1}) if b is a shift of a with a new symbol.
    # Actually, let's use a cleaner construction.
    #
    # Alternative: Use the "cage" construction via expander-like random graphs is too slow.
    # Let's use the known result: the de Bruijn graph B(k, m) as a directed graph has girth related to m.
    # For the undirected version, we can take the underlying undirected graph and remove one direction per edge,
    # but that doesn't give k-regular.
    #
    # Better approach: Use the bipartite double cover or a specific construction.
    #
    # Let me use a simpler, well-known construction:
    # For k=3, g=16: we need girth >= 16.
    # The Lazebnik-Ustimenko-Woldar construction in the seed gives a valid graph but may be large.
    # Let's try to improve it or use a different explicit construction.
    #
    # Actually, let's try the following: construct a k-regular graph using a circulant or generalized Petersen-like
    # structure, but those don't give large girth easily.
    #
    # Let's stick with improving the LUW construction or trying a random approach with verification.
    #
    # Random approach: generate a random k-regular graph with n vertices, check girth.
    # For k=3, g=16, we need n large enough. The Moore bound for (3,16) is 2 + 3*(1+2+4+8+16+32) = 2+3*63 = 191.
    # But we know the answer is around 936, so random graphs of that size might work if we try enough.
    # However, generating and verifying girth for 936-vertex graphs is expensive.
    #
    # Let's try a hybrid: use the LUW construction but optimize it, or try a different explicit family.
    #
    # For now, let's use the LUW construction from the seed but with optimizations.
    if k == 3 and g == 16:
        # Special case: use a known good construction or optimize LUW
        # The LUW construction with q=3, d=11 (since g-5=11, odd, so d=11)
        # gives a 3-regular bipartite graph with girth >= 16.
        # Number of vertices: for LUW CD(3,11), the component size is 2 * 3^11 / something...
        # Actually it's 2 * q^d = 2 * 3^11 = 354294, which is way too large.
        # Wait, the seed says it returns the connected component of the zero point.
        # Let me just run the seed's logic and see.
        
        # Let's try a different approach: use a smaller explicit construction.
        # For (3,16), one known construction uses the Heawood graph iterated or other methods.
        # 
        # Actually, let's try to generate a random 3-regular graph with n vertices and check girth.
        # We'll start with n around 200-400 and increase if needed.
        
        import random
        import sys
        
        def check_girth(n, edges, min_girth):
            """Check if graph with given edges has girth >= min_girth using BFS from each vertex."""
            if n == 0:
                return True
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS from each vertex, tracking parent to avoid backtracking
            for start in range(n):
                # dist[v] = distance, parent[v] = parent in BFS tree
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [start]
                qi = 0
                while qi < len(queue):
                    u = queue[qi]
                    qi += 1
                    if dist[u] >= min_girth // 2:
                        continue  # No need to go further
                    for v in adj[u]:
                        if v == parent[u]:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif dist[v] >= dist[u]:
                            # Cycle found of length dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
            return True
        
        def generate_random_3_reg(n):
            """Generate a random 3-regular graph on n vertices using configuration model."""
            if n < 2 or n % 2 == 1:
                return None
            half_edges = [i for i in range(n) for _ in range(3)]
            random.shuffle(half_edges)
            edges = set()
            for i in range(len(half_edges) // 2):
                u, v = half_edges[2*i], half_edges[2*i+1]
                if u == v:
                    continue
                key = (min(u,v), max(u,v))
                if key in edges:
                    continue
                edges.add(key)
            edge_list = list(edges)
            # Check degree
            deg = [0]*n
            for u,v in edge_list:
                deg[u] += 1
                deg[v] += 1
            if any(d != 3 for d in deg):
                return None
            return edge_list
        
        # Try increasing n
        for n in range(200, 1000, 50):
            if time.time() - t0 > deadline - 10:
                break
            best_edges = None
            for attempt in range(20):
                if time.time() - t0 > deadline - 10:
                    break
                edges = generate_random_3_reg(n)
                if edges is None:
                    continue
                if check_girth(n, edges, g):
                    best_edges = edges
                    break
            if best_edges is not None:
                return n, best_edges
        
        # Fallback: use the LUW construction from the seed
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        table = None
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
    
    else:
        # General case: use LUW construction
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
# EVOLVE-BLOCK-END
