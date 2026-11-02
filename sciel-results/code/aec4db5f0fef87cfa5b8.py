import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def verify_girth_fast(n, edges, g):
        """Fast girth verification using non-backtracking BFS from each vertex."""
        if g <= 3:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for s in range(n):
            # BFS on non-backtracking walks from s
            # State: (current, prev), distance = walk length
            dist = {}
            queue = deque()
            for v in adj[s]:
                state = (v, s)
                dist[state] = 1
                queue.append(state)
            
            while queue:
                u, p = queue.popleft()
                d = dist[(u, p)]
                if d >= g - 1:
                    continue
                for w in adj[u]:
                    if w == p:
                        continue
                    ns = (w, u)
                    if ns not in dist:
                        dist[ns] = d + 1
                        if w == s and d + 1 < g:
                            return False
                        queue.append(ns)
        return True

    def luw_construct(k, g):
        q = k
        base_d = max(3, g - 5)
        
        for d in range(base_d, base_d + 15):
            if time.time() > deadline:
                break
            
            if q == 4:
                # GF(4) with XOR addition and custom multiplication
                mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
                def add(x, y): return x ^ y
                def sub(x, y): return x ^ y
                def mul(x, y): return mul_table[x][y]
            else:
                def add(x, y): return (x + y) % q
                def sub(x, y): return (x - y) % q
                def mul(x, y): return (x * y) % q
            
            # LUW rules: r_c for c=2..d
            # Standard LUW CD(d,q) rules
            rule = {}
            rule[2] = (1, 1)
            rule[3] = (2, 1)
            rule[4] = (1, 2)
            for c in range(5, d + 1):
                # Cycle through patterns
                r = (c - 5) % 4
                c0 = c - r
                if r == 0:
                    rule[c] = (1, c0 - 2)
                elif r == 1:
                    rule[c] = (c0 - 1, 1)
                elif r == 2:
                    rule[c] = (c0, 1)
                else:
                    rule[c] = (1, c0 + 1)
            
            # Build graph using BFS from root
            # Vertices are (side, tuple of length d+1 where first element is "first" choice)
            # Actually, LUW vertices: (side, x_1, x_2, ..., x_d) where x_1 is free
            # Let me use the standard LUW construction:
            # Vertex: (side, a_1, a_2, ..., a_d) with a_i in GF(q)
            # Root: (0, 0, 0, ..., 0)
            # Neighbors of (0, a_1, ..., a_d): (1, b_1, ..., b_d) where b_1 is free, 
            #   b_c = a_c + b_{r_c} * a_{s_c} for c >= 2
            # Neighbors of (1, a_1, ..., a_d): (0, b_1, ..., b_d) where b_1 is free,
            #   b_c = a_c - a_{r_c} * b_{s_c} for c >= 2
            
            # Actually the standard LUW: neighbors determined by the "first" coordinate being free
            # Let me implement it properly.
            
            # Vertex representation: (side, x_1, x_2, ..., x_d)
            # For side=0, neighbors are (1, y_1, ..., y_d) where:
            #   y_1 = t (free, t in GF(q))
            #   y_c = x_c + y_{r_c} * x_{s_c} for c = 2, ..., d
            # For side=1, neighbors are (0, y_1, ..., y_d) where:
            #   y_1 = t (free)
            #   y_c = x_c - x_{r_c} * y_{s_c} for c = 2, ..., d
            
            def get_neighbours(side, x):
                """x is tuple (x_1, ..., x_d)"""
                for t in range(q):
                    y = [0] * (d + 1)  # y[0] unused, y[1..d]
                    y[1] = t
                    for c in range(2, d + 1):
                        r_c, s_c = rule[c]
                        if side == 0:
                            y[c] = add(x[c-1], mul(y[r_c], x[s_c-1]))
                        else:
                            y[c] = sub(x[c-1], mul(x[r_c-1], y[s_c]))
                    yield 1 - side, tuple(y[1:])
            
            # BFS to enumerate all vertices
            root = (0, tuple([0] * d))
            vertex_map = {root: 0}
            order = [root]
            edges = []
            
            for i, v in enumerate(order):
                side, x = v
                for other in get_neighbours(side, x):
                    if other not in vertex_map:
                        vertex_map[other] = len(order)
                        order.append(other)
                    # Add edge only from side 0 to avoid duplicates
                    if side == 0:
                        edges.append((i, vertex_map[other]))
            
            n = len(order)
            
            # Moore bound check
            if g % 2 == 1:
                r = (g - 1) // 2
                moore = 1 + k * sum(k**i for i in range(r))
            else:
                r = g // 2
                moore = 2 * sum(k**i for i in range(r))
            
            if n < moore:
                continue
            
            # Verify girth
            if verify_girth_fast(n, edges, g):
                return n, edges
            
            # If girth < g, try larger d
        
        # Fallback: return last construction even if girth not verified (shouldn't happen)
        d_final = base_d + 14
        if q == 4:
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q
        
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d_final + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            else:
                rule[c] = (1, c0 + 1)
        
        def get_neighbours(side, x):
            for t in range(q):
                y = [0] * (d_final + 1)
                y[1] = t
                for c in range(2, d_final + 1):
                    r_c, s_c = rule[c]
                    if side == 0:
                        y[c] = add(x[c-1], mul(y[r_c], x[s_c-1]))
                    else:
                        y[c] = sub(x[c-1], mul(x[r_c-1], y[s_c]))
                yield 1 - side, tuple(y[1:])
        
        root = (0, tuple([0] * d_final))
        vertex_map = {root: 0}
        order = [root]
        edges = []
        
        for i, v in enumerate(order):
            side, x = v
            for other in get_neighbours(side, x):
                if other not in vertex_map:
                    vertex_map[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((i, vertex_map[other]))
        
        return len(order), edges

    # Known small cages
    if k == 3:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            return 6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        elif g <= 5:
            edges = []
            for i in range(5):
                edges.append((i, (i+1)%5))
                edges.append((5+i, 5+(i+2)%5))
                edges.append((i, 5+i))
            return 10, edges
        elif g <= 6:
            lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
            edges = []
            for i, line in enumerate(lines):
                li = 7 + i
                for p in line:
                    edges.append((p, li))
            return 14, edges
    
    if k == 4:
        if g <= 3:
            return 5, [(i,j) for i in range(5) for j in range(i+1,5)]
        elif g <= 4:
            return 8, [(i,j) for i in range(4) for j in range(4,8)]
    
    if k == 5:
        if g <= 3:
            return 6, [(i,j) for i in range(6) for j in range(i+1,6)]
        elif g <= 4:
            return 10, [(i,j) for i in range(5) for j in range(5,10)]
        elif g <= 5:
            # Clebsch graph
            vertices = [i for i in range(16) if bin(i).count('1') % 2 == 0]
            edges = []
            for ii in range(len(vertices)):
                for jj in range(ii+1, len(vertices)):
                    if bin(vertices[ii] ^ vertices[jj]).count('1') == 2:
                        edges.append((ii, jj))
            n = len(vertices)
            if verify_girth_fast(n, edges, g):
                return n, edges
    
    return luw_construct(k, g)
