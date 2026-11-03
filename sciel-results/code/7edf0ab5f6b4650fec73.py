import math
import sys

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar construction as fallback
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if k != 4 or g != 10:
        return luw_construct(k, g)

    # For k=4, g=10: try to find a smaller graph
    # Strategy: BFS/DFS based search for small 4-regular graphs of girth 10
    # Moore bound for k=4, g=10 (even): 1 + k * sum_{i=0}^{g/2-2} (k-1)^i = 1 + 4*(1 + 3 + 9 + 27) = 1 + 4*40 = 161
    # So we need at least 161 vertices.
    
    # Approach: Use the LPS construction or similar algebraic constructions
    # For (4,10), known constructions include:
    # 1. LUW gives some size
    # 2. LPS (Lubotzky-Phillips-Sarnak) gives Ramanujan graphs
    
    # Let's try LPS construction with p=3: gives a 4-regular graph on 3*(3^2-1) = 24 vertices
    # Girth of LPS(3): The LPS graph with p=3 has girth 4 (too small)
    # p=5: 5*(25-1) = 120 vertices. Girth is at least 2*floor(log_p(5))+2 = 2*1+2 = 4? 
    # Actually girth of LPS(p) is at least 2*log_p(2p) roughly.
    # For p=5: log_5(10) ≈ 1.43, so girth ≈ 2*1.43 ≈ 2.86, so at least 3? No, it's integer.
    # Let me just construct and check.

    def lps_construct(p):
        # LPS graph with prime p
        # Vertices: elements of SL(2, F_p) modulo ±I, so |SL(2,F_p)|/2 = p(p^2-1)/2
        # Wait, standard LPS: vertices are M_2(Z) mod p with det ≡ 1 mod p, i.e., SL(2, F_p) / {±I}
        # Number of vertices: p(p^2-1)/2
        # 4-regular: neighbors via left multiplication by generators
        # Generators: A = [[1,1],[0,1]], B = [[1,0],[1,1]]
        # Actually the 4 generators are:
        # [[1,1],[0,1]], [[1,0],[1,1]], [[1,1],[0,-1]], [[1,0],[-1,1]]
        
        n = p * (p*p - 1) // 2
        
        def mat_mul(a, b):
            return [[(a[0][0]*b[0][0] + a[0][1]*b[1][0]) % p,
                     (a[0][0]*b[0][1] + a[0][1]*b[1][1]) % p],
                    [(a[1][0]*b[0][0] + a[1][1]*b[1][0]) % p,
                     (a[1][0]*b[0][1] + a[1][1]*b[1][1]) % p]]
        
        def mat_eq(a, b):
            return a[0][0] == b[0][0] and a[0][1] == b[0][1] and a[1][0] == b[1][0] and a[1][1] == b[1][1]
        
        def mat_neg(a):
            return [[(-a[0][0]) % p, (-a[0][1]) % p], [(-a[1][0]) % p, (-a[1][1]) % p]]
        
        # Generate all elements of SL(2, F_p)
        elements = []
        seen = set()
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    d = (1 - a*c) % p  # Need det = ad - bc = 1
                    # Actually this doesn't work. Let me enumerate properly.
                    pass
        
        # Better: enumerate all 2x2 matrices with det 1 mod p
        elements = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    # ad - bc = 1 => d = (1 + bc) * a^{-1} mod p if a != 0
                    # If a == 0, then -bc = 1, so bc = -1 mod p
                    if a != 0:
                        a_inv = pow(a, p-2, p)
                        d = (1 + b*c) * a_inv % p
                        M = [[a, b], [c, d]]
                        key = (a, b, c, d)
                        if key not in seen:
                            seen.add(key)
                            elements.append(M)
                    else:
                        # a = 0, so -bc = 1 => bc = p-1
                        if b == 0:
                            continue  # can't have bc = 1 if b = 0
                        # c = (p-1) * b^{-1} mod p
                        b_inv = pow(b, p-2, p)
                        c_val = (p-1) * b_inv % p
                        d = 0  # a = 0, d can be anything? No, det = 0*d - b*c = -bc = 1
                        # So d is determined? No, det = ad - bc = 0 - bc = -bc. We need -bc = 1.
                        # d is free? No, the matrix is [[0, b], [c, d]] and det = -bc.
                        # So d doesn't affect det. But we need all such matrices.
                        # Actually for a=0: det = -bc = 1, so bc = p-1. d is free.
                        # But that gives too many. Let me reconsider.
                        # In SL(2, F_p), a=0 => -bc = 1 => bc = -1 = p-1.
                        # For each b != 0, c = (p-1) * b^{-1}, and d is arbitrary in F_p.
                        for d in range(p):
                            key = (0, b, c_val, d)
                            if key not in seen:
                                seen.add(key)
                                elements.append([[0, b], [c_val, d]])
        
        # Now quotient by ±I: identify M with -M
        # Representative: choose the one with a < p/2 or (a == p//2 and b < p/2) etc.
        # Simpler: just pick one from each pair {M, -M}
        rep_set = set()
        reps = []
        for M in elements:
            key = tuple(M[0]) + tuple(M[1])
            neg_key = tuple(mat_neg(M)[0]) + tuple(mat_neg(M)[1])
            if key not in rep_set and neg_key not in rep_set:
                reps.append(M)
                rep_set.add(key)
                rep_set.add(neg_key)
        
        # Build adjacency
        gen = [
            [[1, 1], [0, 1]],
            [[1, 0], [1, 1]],
            [[1, 1], [0, p-1]],
            [[1, 0], [p-1, 1]]
        ]
        
        elem_index = {tuple(M[0]) + tuple(M[1]): i for i, M in enumerate(reps)}
        n_actual = len(reps)
        edges = []
        for i, M in enumerate(reps):
            for G in gen:
                P = mat_mul(G, M)
                key = tuple(P[0]) + tuple(P[1])
                neg_key = tuple(mat_neg(P)[0]) + tuple(mat_neg(P)[1])
                if key in elem_index:
                    j = elem_index[key]
                elif neg_key in elem_index:
                    j = elem_index[neg_key]
                else:
                    continue
                if i < j:
                    edges.append((i, j))
        return n_actual, edges

    # Check LPS for p = 5, 7, 11, etc.
    for p in [5, 7, 11, 13, 17, 19, 23]:
        n, edges = lps_construct(p)
        if n < 320:
            # Check girth by BFS non-backtracking
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check for cycles of length < g using non-backtracking BFS
            min_cycle = g + 1
            for start in range(n):
                # State: (vertex, prev_vertex, distance)
                # Use BFS from (start, -1, 0)
                visited = set()
                queue = [(start, -1, 0)]
                visited.add((start, -1))
                found = False
                while queue and not found:
                    new_queue = []
                    for u, prev, d in queue:
                        if d >= g:
                            continue
                        for v in adj[u]:
                            if v == prev:
                                continue
                            state = (v, u)
                            if state in visited:
                                # Found a cycle
                                # The cycle length is d + 1 (from start to u in d steps, then u->v where v was visited from some path)
                                # Actually need to be more careful
                                # This simple BFS doesn't directly give cycle length
                                pass
                            else:
                                visited.add(state)
                                if d + 1 < g:
                                    new_queue.append((v, u, d + 1))
                    queue = new_queue
            
            # Simpler girth check: for each vertex, do BFS tracking parent to detect cycles
            # Let's just check if any cycle of length < g exists by checking all-pairs shortest paths
            # with non-backtracking constraint. For small graphs this is fine.
            
            # Actually let's just verify by checking if the graph has girth >= g
            # using a proper non-backtracking BFS from each edge
            has_short_cycle = False
            for u0 in range(n):
                for v0 in adj[u0]:
                    # BFS from edge (u0, v0)
                    # State: (current, prev)
                    dist = {(v0, u0): 1}
                    queue = [(v0, u0)]
                    while queue:
                        new_q = []
                        for cur, prev in queue:
                            d = dist[(cur, prev)]
                            if d >= g:
                                continue
                            for nxt in adj[cur]:
                                if nxt == prev:
                                    continue
                                if nxt == u0:
                                    # Cycle found of length d+1
                                    if d + 1 < g:
                                        has_short_cycle = True
                                        break
                                ns = (nxt, cur)
                                if ns not in dist:
                                    dist[ns] = d + 1
                                    new_q.append(ns)
                        if has_short_cycle:
                            break
                        queue = new_q
                    if has_short_cycle:
                        break
                if has_short_cycle:
                    break
            
            if not has_short_cycle:
                return n, edges
    
    return luw_construct(k, g)
