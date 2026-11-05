import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try LUB graphs for q = 11 to 20
    best_n = float('inf')
    best_edges = None
    
    for q in range(11, 21):
        if time.time() > deadline:
            break
            
        # Generate PSL(2, Z/qZ)
        # Elements are matrices [[a,b],[c,d]] with ad-bc=1 mod q
        # Normalize: identify M and -M. 
        # Canonical form: first non-zero element in row-major order is positive (if q>2)
        # For q=2, it's trivial.
        
        elements = []
        index_map = {}
        
        # Iterate over all SL(2, Z/qZ) matrices
        # a,b,c,d in 0..q-1
        # ad - bc == 1 mod q
        
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # (ad - bc) = 1 => ad = 1 + bc
                    # d = a^{-1} (1 + bc) mod q
                    # a must be invertible
                    if np.gcd(a, q) != 1:
                        continue
                    # Find inverse of a mod q
                    try:
                        a_inv = pow(a, -1, q)
                    except:
                        continue
                    for d in range(q):
                        # Check ad - bc == 1
                        if (a*d - b*c) % q != 1:
                            continue
                        # Matrix M = [[a,b],[c,d]]
                        # Normalize
                        # If a==0 and b==0 and c==0: impossible (det=1)
                        # Find first non-zero
                        # Order: a, b, c, d
                        if a != 0:
                            # Make a positive? In Z/qZ, "positive" is ambiguous.
                            # We can just use the tuple (a,b,c,d) and (q-a, q-b, q-c, q-d)
                            # Pick the lexicographically smaller one
                            m1 = (a, b, c, d)
                            m2 = (q-a, q-b, q-c, q-d)
                            if m1 < m2:
                                key = m1
                            else:
                                key = m2
                        elif b != 0:
                            m1 = (a, b, c, d)
                            m2 = (q-a, q-b, q-c, q-d)
                            if m1 < m2:
                                key = m1
                            else:
                                key = m2
                        elif c != 0:
                            m1 = (a, b, c, d)
                            m2 = (q-a, q-b, q-c, q-d)
                            if m1 < m2:
                                key = m1
                            else:
                                key = m2
                        else:
                            # d != 0
                            m1 = (a, b, c, d)
                            m2 = (q-a, q-b, q-c, q-d)
                            if m1 < m2:
                                key = m1
                            else:
                                key = m2
                        
                        if key not in index_map:
                            index_map[key] = len(elements)
                            elements.append(key)
        
        n = len(elements)
        if n < 512:
            continue
            
        # Build graph
        # Generators in PSL:
        # S = [[0, -1], [1, 0]] -> [[0, q-1], [1, 0]]
        # T = [[1, 1], [0, 1]]
        # T_inv = [[1, -1], [0, 1]] -> [[1, q-1], [0, 1]]
        
        S = (0, q-1, 1, 0)
        T = (1, 1, 0, 1)
        T_inv = (1, q-1, 0, 1)
        
        # Map generators to normalized keys
        def normalize(m):
            a,b,c,d = m
            m1 = (a,b,c,d)
            m2 = ((q-a)%q, (q-b)%q, (q-c)%q, (q-d)%q)
            return m1 if m1 < m2 else m2
            
        S_key = normalize(S)
        T_key = normalize(T)
        T_inv_key = normalize(T_inv)
        
        # Check if generators are in the group (they should be)
        if S_key not in index_map or T_key not in index_map or T_inv_key not in index_map:
            continue
            
        # Build adjacency
        # For each element, multiply by S, T, T_inv
        # Multiplication: [[a,b],[c,d]] * [[e,f],[g,h]] = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        adj = [[] for _ in range(n)]
        
        for i, m in enumerate(elements):
            a,b,c,d = m
            # Multiply by S
            # S = [[0, q-1], [1, 0]]
            # M*S = [[a*0 + b*1, a*(q-1) + b*0], [c*0 + d*1, c*(q-1) + d*0]]
            # = [[b, a*(q-1)], [d, c*(q-1)]]
            p1 = (b % q, (a*(q-1)) % q, d % q, (c*(q-1)) % q)
            p1_key = normalize(p1)
            j1 = index_map[p1_key]
            adj[i].append(j1)
            
            # Multiply by T
            # T = [[1, 1], [0, 1]]
            # M*T = [[a*1 + b*0, a*1 + b*1], [c*1 + d*0, c*1 + d*1]]
            # = [[a, a+b], [c, c+d]]
            p2 = (a % q, (a+b) % q, c % q, (c+d) % q)
            p2_key = normalize(p2)
            j2 = index_map[p2_key]
            adj[i].append(j2)
            
            # Multiply by T_inv
            # T_inv = [[1, q-1], [0, 1]]
            # M*T_inv = [[a*1 + b*0, a*(q-1) + b*1], [c*1 + d*0, c*(q-1) + d*1]]
            # = [[a, a*(q-1)+b], [c, c*(q-1)+d]]
            p3 = (a % q, (a*(q-1) + b) % q, c % q, (c*(q-1) + d) % q)
            p3_key = normalize(p3)
            j3 = index_map[p3_key]
            adj[i].append(j3)
            
        # Check girth
        # BFS from each node to find shortest cycle
        # Or just BFS from one node? No, girth is min over all cycles.
        # We can do BFS from each node, but it's expensive.
        # However, n is small (~1000).
        # We can check if there is a cycle of length < g.
        
        has_short_cycle = False
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            q_bfs = deque()
            q_bfs.append(start)
            dist[start] = 0
            parent[start] = -1
            
            while q_bfs:
                u = q_bfs.popleft()
                if dist[u] >= g:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q_bfs.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But this is not necessarily the shortest cycle through u,v
                        # Actually, if we find a back edge, the cycle length is dist[u] + dist[v] + 1
                        # This works for finding *a* cycle. To find the *girth*, we need the minimum.
                        # Since we break at dist >= g, if we find a cycle of length < g, we return False.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            has_short_cycle = True
                            break
            if has_short_cycle:
                break
                
        if not has_short_cycle:
            # Girth >= g
            if n < best_n:
                best_n = n
                # Construct edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_edges = edges
                # If we found a very small graph, we can return early?
                # But maybe a larger q gives smaller n? No, n grows with q.
                # So the first one found (smallest q) is the best.
                # Wait, q=11 gives n=660. q=12 gives n=858.
                # So we should return the first valid one found.
                return n, edges
                
    # If no LUB graph found, fallback to seed logic?
    # The seed is huge.
    # We must return something valid.
    # If we didn't find any, we return a dummy or the seed.
    # But the seed is valid.
    # Let's just return the seed if we fail.
    # However, the seed code is in the prompt. I can copy it.
    
    # Re-implementing seed for fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
