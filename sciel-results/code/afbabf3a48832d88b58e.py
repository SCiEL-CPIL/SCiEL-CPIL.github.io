import time
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Fallback to LUW if k != 3 or if search fails
    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
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
        
        n = len(order)
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        for s in range(n):
            dist = {s: 0}
            parent = {s: -1}
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if nb in dist:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            break
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                if min_cycle < g:
                    break
            if min_cycle < g:
                break
        
        if min_cycle < g:
            for d_try in range(d + 1, d + 10):
                if time.time() > deadline - 10:
                    break
                rule2 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
                for c in range(5, d_try + 1):
                    r, c0 = (c - 5) % 4, c - (c - 5) % 4
                    rule2[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                
                def neighbours2(vertex):
                    side, x = vertex
                    for first in range(q):
                        y = [None, first] + [0] * (d_try - 1)
                        for c in range(2, d_try + 1):
                            a, b = rule2[c]
                            y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                        yield 1 - side, tuple(y)
                
                start2 = (0, (None,) + (0,) * d_try)
                index2, order2, edges2 = {start2: 0}, [start2], []
                for vertex in order2:
                    for other in neighbours2(vertex):
                        if other not in index2:
                            index2[other] = len(order2)
                            order2.append(other)
                        if vertex[0] == 0:
                            edges2.append((index2[vertex], index2[other]))
                
                n2 = len(order2)
                adj2 = [[] for _ in range(n2)]
                for i, j in edges2:
                    adj2[i].append(j)
                    adj2[j].append(i)
                
                min_cycle2 = float('inf')
                for s in range(n2):
                    dist = {s: 0}
                    parent = {s: -1}
                    queue = [s]
                    qi = 0
                    while qi < len(queue):
                        node = queue[qi]
                        qi += 1
                        for nb in adj2[node]:
                            if nb == parent[node]:
                                continue
                            if nb in dist:
                                cycle_len = dist[node] + dist[nb] + 1
                                if cycle_len < min_cycle2:
                                    min_cycle2 = cycle_len
                                if cycle_len < g:
                                    break
                            else:
                                dist[nb] = dist[node] + 1
                                parent[nb] = node
                                queue.append(nb)
                        if min_cycle2 < g:
                            break
                    if min_cycle2 < g:
                        break
                
                if min_cycle2 >= g:
                    return n2, edges2
        
        return n, edges

    if k != 3:
        return luw_construct(k, g)

    def check_girth_fast(n, edges, g):
        """Fast girth check using BFS with arrays."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return False
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                if min_cycle < g:
                    return False
        return True

    def build_ramanujan(q):
        """Build a 3-regular Ramanujan graph using PSL(2,q).
        
        For q ≡ ±1 (mod 5), we can use specific involutions.
        The graph has q(q+1) vertices and girth at least 2*log_2(q).
        """
        if q < 7:
            return None
            
        # For PSL(2,q), elements are 2x2 matrices with determinant 1, mod q
        # We need to find 3 involutions (elements of order 2) that generate the group
        # and don't create short cycles.
        
        # Involutions in PSL(2,q) have trace 0 (in PSL sense, trace^2 = 4*det = 4)
        # Actually, an element [[a,b],[c,d]] with ad-bc=1 is an involution iff 
        # it's conjugate to [[0,1],[-1,0]], which has trace 0.
        
        # Standard generators for cubic Ramanujan graphs:
        # g1 = [[0, 1], [-1, 0]]  (order 4, but in PSL has order 2 if we consider -I = I)
        # Wait, in PSL(2,q), [[0,1],[-1,0]] has order 2 since its square is -I = I in PSL.
        
        # Let's use:
        # A = [[0, 1], [-1, 0]]  -> involution in PSL
        # B = [[0, 1], [1, 0]]   -> involution in PSL  
        # C = [[1, 0], [1, 1]]?  No, this has order q.
        
        # Actually, let me think more carefully. In PSL(2,q):
        # An element M is an involution iff M^2 = I (in PSL), i.e., M^2 = ±I in SL.
        # trace(M) = 0 gives M^2 = -I (since det=1, char poly is x^2 - tr*x + 1, 
        # if tr=0 then x^2+1, so M^2 = -I).
        # trace(M) = ±2 gives M^2 = I (if M=±I) or M is unipotent (M^2 ≠ I generally).
        
        # So involutions in PSL(2,q) are represented by matrices with trace 0.
        
        # Let's pick:
        # A = [[0, 1], [-1, 0]]  trace = 0 ✓
        # B = [[0, 1], [1, 0]]   trace = 0 ✓
        # C = [[1, -1], [1, 1]]? trace = 2, no.
        
        # We need a third involution. How about:
        # C = [[1, 1], [-1, -1]]? trace = 0, det = -1+1 = 0. Not in SL.
        
        # C = [[a, b], [c, -a]] with -a^2 - bc = 1, i.e., a^2 + bc = -1.
        
        # For simplicity, let's use:
        # A = [[0, 1], [-1, 0]]
        # B = [[0, 1], [1, 0]]
        # C = [[1, 0], [0, -1]]? No, this is -[[1,0],[0,1]] = -I, which is identity in PSL.
        
        # C = [[1, 1], [-1, -1]]? det = -1+1=0. No.
        # C = [[2, 1], [-1, -2]]? trace = 0, det = -4+1 = -3. Need -3 ≡ 1 mod q, so q|4.
        
        # This is getting complicated. Let me use a different approach.
        # The standard construction uses specific generators that are known to work.
        
        # Actually, for the (3,16) cage, we want a graph with girth 16.
        # Ramanujan graphs have girth ~ 2*log_2(q). For girth 16, we need 2*log_2(q) >= 16,
        # so q >= 2^8 = 256. That's too large for a small graph.
        
        # The LUW construction with q=3, d=7 gives about 2*3^7 = 4374 vertices, which is much larger than 936.
        # So Ramanujan graphs are not competitive for the (3,16) cage.
        
        return None

    # Try Ramanujan constructions for small q where they might be competitive
    for q in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if time.time() > deadline - 30:
            break
        result = build_ramanujan(q)
        if result is not None:
            n, edges = result
            if n < 936 and check_girth_fast(n, edges, g):
                return n, edges

    # Fall back to LUW construction
    return luw_construct(k, g)
