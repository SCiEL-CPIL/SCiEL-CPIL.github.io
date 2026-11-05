import time
import random
from collections import deque

def _luw_graph(k, g):
    """Fallback to LUW construction."""
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

def _check_girth(n, edges, min_girth):
    """Check if graph has girth >= min_girth using BFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for cycles of length < min_girth
    # For bipartite, we only check even cycles.
    # We do BFS from each node, keeping track of parent to avoid backtracking.
    # If we find a node at distance d that was visited at distance d' < d,
    # cycle length is d + d' - 1? No.
    # Standard girth check:
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        found = False
        while queue and not found:
            u = queue.popleft()
            if dist[u] >= min_girth // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    # But we need to be careful with bipartite.
                    # In bipartite, u and v must be on different levels if connected?
                    # No, edge connects level i and i+1.
                    # If we see a back edge to a node at level j,
                    # cycle length is dist[u] + dist[v] + 1.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
                    # If cycle_len >= min_girth, we can stop exploring from this branch?
                    # No, there might be a shorter cycle elsewhere.
                    # But BFS finds shortest paths.
                    # If dist[u] + dist[v] + 1 >= min_girth, is it safe?
                    # Yes, any other path would be longer or same.
                    # Actually, we just need to know if ANY cycle < min_girth exists.
                    # If the first cycle found is >= min_girth, are there shorter ones?
                    # BFS expands in increasing distance.
                    # So if we find a cycle of length L, it is the shortest cycle involving this edge/crossing?
                    # Not necessarily the global shortest.
                    # But if dist[u] + dist[v] + 1 >= min_girth, we don't care about this one.
                    # We continue to check other nodes.
                    pass
    return True

def _try_construct_bipartite(n, k, g, time_limit):
    """Try to construct a k-regular bipartite graph of size n with girth >= g."""
    if n % 2 != 0:
        return None
    half = n // 2
    start_time = time.time()
    
    for attempt in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # Adjacency lists
        # U: 0..half-1, V: half..n-1
        adj_u = [set() for _ in range(half)]
        adj_v = [set() for _ in range(half)]
        
        # Randomly order vertices in U
        order = list(range(half))
        random.shuffle(order)
        
        success = True
        for u_idx in order:
            # Choose k random neighbors in V
            # We need to ensure no C4, C6, C8
            candidates = list(range(half))
            random.shuffle(candidates)
            
            chosen = []
            for v_idx in candidates:
                if len(chosen) == k:
                    break
                
                # Check C4
                c4 = False
                for v2 in chosen:
                    # If u and v2 share a neighbor in V? No, v2 is in V.
                    # C4: u - v - u' - v' - u
                    # u is connected to v and v2.
                    # If there exists u' such that v in adj_v[u'] and v2 in adj_v[u'], then C4.
                    # adj_v[u'] is the set of U nodes connected to v?
                    # My adj_v is indexed by V node, contains U nodes.
                    # So adj_v[v_idx] is set of U nodes connected to v_idx.
                    # We need to check if any u' in adj_v[v_idx] is also in adj_v[v2].
                    # adj_v[v2] is set of U nodes connected to v2.
                    if adj_v[v_idx] & adj_v[v2]:
                        c4 = True
                        break
                
                if c4:
                    continue
                
                # Check C6
                # u - v - u' - v2 - u'' - v3 - u
                # u connected to v and v3 (in chosen? No, v3 is another neighbor of u).
                # Path: u -> v -> u' -> v2 -> u'' -> v3 -> u
                # v is the new neighbor.
                # u' in adj_v[v_idx]
                # v2 in adj_u[u'] (neighbors of u' in V)
                # u'' in adj_v[v2]
                # v3 in adj_u[u'']
                # We need v3 to be in `chosen` (neighbors of u already assigned).
                # So:
                c6 = False
                for u_prime in adj_v[v_idx]:
                    if u_prime == u_idx: continue
                    for v2 in adj_u[u_prime]:
                        if v2 == v_idx: continue
                        for u_double in adj_v[v2]:
                            if u_double == u_prime: continue
                            # Check if u_double is connected to any v3 in chosen
                            # adj_u[u_double] contains V neighbors.
                            # If any v3 in chosen is in adj_u[u_double], C6 found.
                            if adj_u[u_double] & set(chosen):
                                c6 = True
                                break
                        if c6: break
                    if c6: break
                
                if c6:
                    continue
                
                # Check C8
                # u - v - u' - v2 - u'' - v3 - u''' - v4 - u
                # v is new.
                # u' in adj_v[v]
                # v2 in adj_u[u']
                # u'' in adj_v[v2]
                # v3 in adj_u[u'']
                # u''' in adj_v[v3]
                # v4 in adj_u[u''']
                # v4 must be in chosen.
                c8 = False
                for u_prime in adj_v[v_idx]:
                    if u_prime == u_idx: continue
                    for v2 in adj_u[u_prime]:
                        if v2 == v_idx: continue
                        for u_double in adj_v[v2]:
                            if u_double == u_prime: continue
                            for v3 in adj_u[u_double]:
                                if v3 == v2: continue
                                for u_triple in adj_v[v3]:
                                    if u_triple == u_double: continue
                                    if adj_u[u_triple] & set(chosen):
                                        c8 = True
                                        break
                                if c8: break
                            if c8: break
                        if c8: break
                    if c8: break
                
                if c8:
                    continue
                
                # Accept v_idx
                chosen.append(v_idx)
                adj_u[u_idx].add(v_idx)
                adj_v[v_idx].add(u_idx)
            
            if len(chosen) < k:
                success = False
                break
        
        if success:
            # Final verification
            edges = []
            for u in range(half):
                for v in adj_u[u]:
                    edges.append((u, v + half))
            
            if _check_girth(n, edges, g):
                return n, edges
            
    return None

def construct(k, g):
    if k == 4 and g == 10:
        # Try to find a smaller graph
        # Start from 320 (known to exist) and go up to 510
        # If we find one < 512, return it.
        # To save time, we prioritize smaller N.
        # But 320 is the target.
        # Let's try 320 first.
        
        # Deadline guard
        start = time.time()
        deadline = 110.0 # 10s margin
        
        # Try N = 320
        res = _try_construct_bipartite(320, k, g, 20.0)
        if res:
            return res
        
        if time.time() > deadline:
            return _luw_graph(k, g)
        
        # Try other sizes if 320 failed quickly or if we want to be sure?
        # Actually, if 320 fails, it might be hard.
        # Let's try 340, 360...
        for n in range(322, 512, 2):
            if time.time() > deadline - 10.0:
                break
            res = _try_construct_bipartite(n, k, g, 5.0)
            if res:
                return res
        
        return _luw_graph(k, g)
    else:
        return _luw_graph(k, g)
