import time
import random
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Random voltage assignment search over Petersen graph for k=3
    def petersen_voltage_search(k, g):
        # Petersen graph: 10 vertices (0-9), 15 edges, girth 5
        petersen_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner pentagram
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
        ]
        
        n_base = 10
        n_edges = len(petersen_edges)
        
        # Edge index lookup
        edge_idx = {}
        for i, (u, v) in enumerate(petersen_edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
        
        # Build base adjacency
        base_adj = [[] for _ in range(n_base)]
        for (u, v) in petersen_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)
        
        best_result = None
        best_n = float('inf')
        
        for m in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]:
            if time.time() > deadline - 5:
                break
            
            n_total = n_base * m
            
            attempts = 0
            max_attempts = 50000
            
            while attempts < max_attempts and time.time() < deadline - 10:
                attempts += 1
                
                voltages = [random.randint(0, m-1) for _ in range(n_edges)]
                
                adj = [[] for _ in range(n_total)]
                edges = []
                
                for (u, v) in petersen_edges:
                    eidx = edge_idx[(u, v)]
                    vol = voltages[eidx]
                    for t in range(m):
                        u_idx = u * m + t
                        v_idx = v * m + ((t + vol) % m)
                        adj[u_idx].append(v_idx)
                        adj[v_idx].append(u_idx)
                        edges.append((u_idx, v_idx))
                
                # Check girth using non-backtracking walks
                girth_ok = True
                
                for s in range(n_total):
                    if not girth_ok:
                        break
                    
                    visited = set()
                    queue = [(s, -1)]
                    visited.add((s, -1))
                    depth = 0
                    
                    while queue and depth < g - 1:
                        new_queue = []
                        for (node, prev) in queue:
                            for nb in adj[node]:
                                if nb == prev:
                                    continue
                                if nb == s and depth + 1 >= 3:
                                    girth_ok = False
                                    break
                                if (nb, node) not in visited:
                                    visited.add((nb, node))
                                    new_queue.append((nb, node))
                            if not girth_ok:
                                break
                        queue = new_queue
                        depth += 1
                
                if girth_ok:
                    edge_set = set()
                    valid = True
                    for (u, v) in edges:
                        if u == v:
                            valid = False
                            break
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            valid = False
                            break
                        edge_set.add(key)
                    
                    if valid:
                        deg = [0] * n_total
                        for (u, v) in edges:
                            deg[u] += 1
                            deg[v] += 1
                        if all(d == k for d in deg):
                            if n_total < best_n:
                                best_n = n_total
                                best_result = (n_total, edges)
                                break
            
            if best_result is not None:
                break
        
        return best_result
    
    # Fallback to LUW if k != 3 or if search fails
    def luw_construct(k, g):
        q = k
        # For LUW construction, girth = 2*d + 2, so d = (g - 2) // 2
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
        
        # Verify girth using BFS from each vertex
        n = len(order)
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Check girth: find shortest cycle via BFS from each vertex
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
            # Girth too small, increase d and retry
            for d_try in range(d + 1, d + 10):
                if time.time() > deadline - 10:
                    break
                # Reconstruct with larger d
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

    # Try random voltage assignment over Petersen graph
    result = petersen_voltage_search(k, g)
    if result is not None:
        return result
    
    # Fallback to LUW
    return luw_construct(k, g)
