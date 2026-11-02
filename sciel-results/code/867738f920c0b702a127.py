import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; keep margin before 150s kill

    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
        q = k
        # Ensure d is sufficient for girth g
        # g = 2d + 2 roughly
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
            # The rule generation from the seed
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

    # Algebraic search for smaller graphs
    def search_algebraic(k, g):
        if k != 3:
            return None
        
        best_result = None
        best_n = float('inf')
        
        # Search for m such that m*m is a candidate
        # Start from small m to find small n
        # Moore bound for (3, 16) is 512. 
        # Try m from 23 (529) up to 40 (1600)
        for m in range(23, 41):
            if time.time() - t0 > deadline - 10:
                break
            
            n = m * m
            if n < 512:
                continue
                
            # Try random linear functions
            # f1(u) = a*u + b1, f2(u) = c*u + b2
            # s is the shift for the third edge
            
            trials_per_m = 500 if m > 30 else 1000
            for trial in range(trials_per_m):
                if time.time() - t0 > deadline - 5:
                    break
                
                a = random.randint(1, m - 1)
                b1 = random.randint(0, m - 1)
                c = random.randint(1, m - 1)
                b2 = random.randint(0, m - 1)
                s = random.randint(1, m - 1)
                
                # Pre-compute neighbors for girth check
                # adj[i] = [u1, u2, u3]
                adj = [[0, 0, 0] for _ in range(n)]
                valid_graph = True
                
                for u in range(m):
                    f1_u = (a * u + b1) % m
                    f2_u = (c * u + b2) % m
                    for v in range(m):
                        idx = u * m + v
                        u1 = (u + 1) % m
                        v1 = (v + f1_u) % m
                        adj[idx][0] = u1 * m + v1
                        
                        u2 = (u - 1) % m
                        v2 = (v + f2_u) % m
                        adj[idx][1] = u2 * m + v2
                        
                        v3 = (v + s) % m
                        adj[idx][2] = u * m + v3
                        
                        # Check for self-loops
                        if adj[idx][0] == idx or adj[idx][1] == idx or adj[idx][2] == idx:
                            valid_graph = False
                            break
                    if not valid_graph:
                        break
                if not valid_graph:
                    continue
                    
                # Check for multi-edges (simple graph check)
                # For undirected, we just need to ensure neighbors are distinct
                for i in range(n):
                    if len(set(adj[i])) != 3:
                        valid_graph = False
                        break
                if not valid_graph:
                    continue
                    
                # Check Girth
                # BFS from every vertex
                girth_ok = True
                half_girth = g // 2
                
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    qi = 0
                    while qi < len(queue):
                        u = queue[qi]
                        qi += 1
                        if dist[u] >= half_girth:
                            continue
                        for v in adj[u]:
                            if v == parent[u]:
                                continue
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            else:
                                # Cycle found
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < g:
                                    girth_ok = False
                                    break
                        if not girth_ok:
                            break
                    if not girth_ok:
                        break
                        
                if girth_ok:
                    # Construct edge list
                    edges = []
                    seen = set()
                    for u in range(m):
                        f1_u = (a * u + b1) % m
                        f2_u = (c * u + b2) % m
                        for v in range(m):
                            idx = u * m + v
                            # Edge 1
                            u1 = (u + 1) % m
                            v1 = (v + f1_u) % m
                            nidx = u1 * m + v1
                            key = (min(idx, nidx), max(idx, nidx))
                            if key not in seen:
                                seen.add(key)
                                edges.append((idx, nidx))
                            # Edge 2
                            u2 = (u - 1) % m
                            v2 = (v + f2_u) % m
                            nidx = u2 * m + v2
                            key = (min(idx, nidx), max(idx, nidx))
                            if key not in seen:
                                seen.add(key)
                                edges.append((idx, nidx))
                            # Edge 3
                            v3 = (v + s) % m
                            nidx = u * m + v3
                            key = (min(idx, nidx), max(idx, nidx))
                            if key not in seen:
                                seen.add(key)
                                edges.append((idx, nidx))
                    
                    # Final verification of regularity and simplicity
                    deg = [0] * n
                    for a_e, b_e in edges:
                        deg[a_e] += 1
                        deg[b_e] += 1
                    if all(d == 3 for d in deg) and len(edges) == n * 3 // 2:
                        if n < best_n:
                            best_n = n
                            best_result = (n, edges)
                            # We found a valid graph. Return immediately for smallest m?
                            # Since we iterate m increasing, the first found is likely smallest n.
                            # However, a larger m might have a valid graph if smaller m fails.
                            # But we want MINIMUM n. So if we find one for m=23 (529), that's great.
                            # If we find one for m=31 (961), it's better than LUW (300k).
                            # We should keep searching for smaller m if we haven't tried all?
                            # We ARE iterating m increasing. So the first success is the smallest m success.
                            return best_result
                            
        return best_result

    # Main logic
    if k == 3 and g == 16:
        # Try Algebraic construction first
        alg_result = search_algebraic(k, g)
        if alg_result is not None:
            return alg_result
        
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
# EVOLVE-BLOCK-END
