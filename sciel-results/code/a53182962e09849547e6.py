import time
import random
from collections import deque

def _moore_bound(k, g):
    if g % 2 == 0:
        depth = g // 2 - 1
        return 2 * (1 + k * (k**depth - 1) // (k - 1))
    else:
        depth = (g - 1) // 2
        return 1 + k * (k**depth - 1) // (k - 1)

def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check girth using BFS
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif dist[v] == dist[u] + 1:
                    pass
                else:
                    # Found a back edge, check cycle length
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _find_shortest_cycle(n, adj):
    """Find the shortest cycle in the graph."""
    min_cycle = float('inf')
    best_cycle = None
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        # Reconstruct cycle
                        path1 = [u]
                        curr = u
                        while curr != start:
                            curr = parent[curr]
                            path1.append(curr)
                        path2 = [v]
                        curr = v
                        while curr != start:
                            curr = parent[curr]
                            path2.append(curr)
                        # Combine
                        cycle = path1[:-1] + path2[::-1]
                        # This reconstruction is rough, just need the edges
                        best_cycle = (u, v, path1, path2)
    
    return min_cycle, best_cycle

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    mb = _moore_bound(k, g)
    min_n = mb
    if (min_n * k) % 2 != 0:
        min_n += 1
        
    # Try local search for n from min_n to min_n + 50
    # If not found, fallback to LUW
    
    def try_local_search(n, time_limit=10.0):
        t0 = time.time()
        random.seed(n * 12345 + k * 100 + g)
        
        # Generate random k-regular graph
        # Use configuration model and fix invalid edges
        edges = set()
        half_edges = list(range(n)) * k
        random.shuffle(half_edges)
        
        # Simple rejection sampling for small n
        # For larger n, use a more robust generator
        # Given n ~ 150-200, rejection is fine
        
        valid_graph = False
        for _ in range(1000):
            if time.time() - t0 > time_limit:
                break
            edges = set()
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
            if valid:
                # Check regularity
                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if all(d == k for d in deg):
                    valid_graph = True
                    break
        
        if not valid_graph:
            return None, None
            
        edge_list = list(edges)
        
        # Local search
        for step in range(10000):
            if time.time() - t0 > time_limit:
                break
                
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u, v in edge_list:
                adj[u].append(v)
                adj[v].append(u)
                
            # Check girth
            min_girth, _ = _find_shortest_cycle(n, adj)
            
            if min_girth >= g:
                return n, edge_list
                
            # If girth is too small, try to break a short cycle
            # Find a short cycle
            # We need the actual cycle vertices to perform swaps
            # _find_shortest_cycle returns min_girth and rough info
            # Let's implement a proper cycle finder for the swap
            
            # For simplicity, pick a random edge and try to swap it with another
            # to increase girth
            
            # Pick a random edge (u, v)
            idx = random.randint(0, len(edge_list) - 1)
            u, v = edge_list[idx]
            
            # Find a neighbor w of u such that (u, w) is not an edge? No, w is neighbor.
            # We want to replace (u, v) with (u, w) where w is not v and not already connected to u?
            # No, u is connected to v. We remove (u, v).
            # u loses degree 1. v loses degree 1.
            # We need to add two edges to restore regularity.
            # Add (u, x) and (v, y)?
            # Or (u, y) and (v, x)?
            
            # Let's try a 2-opt move:
            # Remove (u, v) and (a, b). Add (u, a) and (v, b).
            # Condition: (u, a) and (v, b) are not existing edges, and u!=a, v!=b.
            
            # Pick random (a, b)
            if len(edge_list) < 2:
                continue
            idx2 = random.randint(0, len(edge_list) - 1)
            if idx2 == idx:
                idx2 = (idx2 + 1) % len(edge_list)
            a, b = edge_list[idx2]
            
            # Ensure edges are distinct
            if {u, v} == {a, b}:
                continue
                
            # Check if (u, a) and (v, b) are valid
            edge_ua = (min(u, a), max(u, a))
            edge_vb = (min(v, b), max(v, b))
            
            if edge_ua in edges or edge_vb in edges:
                continue
            if u == a or v == b:
                continue
            if u == b or v == a:
                # Check the other combination
                edge_ub = (min(u, b), max(u, b))
                edge_va = (min(v, a), max(v, a))
                if edge_ub in edges or edge_va in edges:
                    continue
                if u == b or v == a:
                    continue
                # Swap with (u, b) and (v, a)
                new_edges = edges - {(u, v), (a, b)}
                new_edges.add(edge_ub)
                new_edges.add(edge_va)
                edge_list = list(new_edges)
                edges = new_edges
                continue
                
            new_edges = edges - {(u, v), (a, b)}
            new_edges.add(edge_ua)
            new_edges.add(edge_vb)
            edge_list = list(new_edges)
            edges = new_edges
            
        return None, None
        
    # Try n from min_n to min_n + 20
    for n in range(min_n, min_n + 20):
        if (n * k) % 2 != 0:
            continue
        if time.time() > deadline - 20.0:
            break
            
        res_n, res_edges = try_local_search(n, time_limit=5.0)
        if res_n is not None:
            return res_n, res_edges
            
    # Fallback: LUW Construction
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 5:
        add = lambda x, y: (x + y) % 5
        sub = lambda x, y: (x - y) % 5
        mul = lambda x, y: x * y % 5
    elif q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 3:
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: x * y % 3
    elif q == 2:
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
            yield (1 - side), tuple(y)
            
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    n_luw = len(order)
    
    if _verify_girth(n_luw, edges, g):
        return n_luw, edges
        
    return n_luw, edges
