import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 140.0
    
    # LUW construction as fallback
    def luw_construct(k, g):
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
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
    
    # Fast girth verification
    def verify_girth(n, edges, g):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Non-backtracking BFS from each vertex
        for start in range(n):
            dist = {}
            queue = deque()
            for nb in adj[start]:
                dist[(start, nb)] = 1
                queue.append((start, nb))
            while queue:
                u, v = queue.popleft()
                d = dist[(u, v)]
                if d >= g - 1:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == start:
                        if d + 1 < g:
                            return False
                        continue
                    if (v, w) not in dist:
                        dist[(v, w)] = d + 1
                        queue.append((v, w))
        return True
    
    # For (3,16), try PSL(2,13) Cayley graph search
    if k == 3 and g == 16:
        # First get LUW baseline
        n_luw, edges_luw = luw_construct(k, g)
        best_n, best_edges = n_luw, edges_luw
        
        # Try PSL(2,13)
        try:
            q = 13
            # Build SL(2,q)
            sl2 = []
            sl2_set = set()
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        for d in range(q):
                            if (a * d - b * c) % q == 1:
                                e = (a, b, c, d)
                                if e not in sl2_set:
                                    sl2_set.add(e)
                                    sl2.append(e)
            
            # PSL(2,q) = SL(2,q) / {±I}
            psl = {}
            for e in sl2:
                a, b, c, d = e
                inv = (d, -b % q, -c % q, a)
                key = min(e, inv)
                psl[key] = e
            
            n_psl = len(psl)
            elem_list = list(psl.keys())
            elem_idx = {e: i for i, e in enumerate(elem_list)}
            
            def mat_mult(m1, m2):
                a1, b1, c1, d1 = m1
                a2, b2, c2, d2 = m2
                return ((a1*a2 + b1*c2) % q, (a1*b2 + b1*d2) % q,
                        (c1*a2 + d1*c2) % q, (c1*b2 + d1*d2) % q)
            
            # Build multiplication table
            mult = [[0] * n_psl for _ in range(n_psl)]
            for i, e1 in enumerate(elem_list):
                for j, e2 in enumerate(elem_list):
                    prod = mat_mult(e1, e2)
                    # Find canonical representative
                    a, b, c, d = prod
                    inv = (d, -b % q, -c % q, a)
                    key = min(prod, inv)
                    mult[i][j] = elem_idx[key]
            
            # Find involutions
            identity = (1, 0, 0, 1)
            involutions = []
            for i, e in enumerate(elem_list):
                if e == identity:
                    continue
                if mult[i][i] == 0:  # e^2 = identity
                    involutions.append(i)
            
            # Try random Cayley graphs: {a, a^{-1}, b} where b is involution
            rng = random.Random(42)
            found = None
            
            for attempt in range(200):
                if time.time() > deadline - 10.0:
                    break
                
                # Pick a non-involution element
                non_inv = [i for i in range(n_psl) if i not in set(involutions) and i != 0]
                if not non_inv:
                    break
                a = rng.choice(non_inv)
                a_inv = mult[a][0]  # a * identity = a, need inverse
                # Find inverse properly
                for j in range(n_psl):
                    if mult[a][j] == 0:
                        a_inv = j
                        break
                
                if a == a_inv:
                    continue
                
                b = rng.choice(involutions)
                if b in (a, a_inv):
                    continue
                
                gens = [a, a_inv, b]
                if len(set(gens)) != 3:
                    continue
                
                # Build graph
                adj = [set() for _ in range(n_psl)]
                valid = True
                for i in range(n_psl):
                    for gi in gens:
                        j = mult[i][gi]
                        adj[i].add(j)
                        adj[j].add(i)
                    if len(adj[i]) != 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check girth
                if verify_girth(n_psl, [(i, j) for i in range(n_psl) for j in adj[i] if i < j], g):
                    found = (n_psl, [(i, j) for i in range(n_psl) for j in adj[i] if i < j])
                    break
            
            if found:
                best_n, best_edges = found
        
        except Exception:
            pass
        
        return best_n, best_edges
    
    return luw_construct(k, g)
