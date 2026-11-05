import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    def check_girth_bfs(adj, n, g):
        """Check girth using BFS from each vertex - no cycle shorter than g."""
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        if dist[x] + dist[y] + 1 < g:
                            return False
        return True

    def is_connected(adj, n):
        if n == 0:
            return True
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return None
        if not is_connected(adj, n):
            return None
        if not check_girth_bfs(adj, n, g):
            return None
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- LUW Construction ----------
    # For k=5, g=9 (odd, d=4): Use the Lazebnik-Ustimenko-Woldar construction
    # Vertices: (side, x1, x2, x3, x4) with side in {0,1}, xi in Z_5
    # n = 2 * 5^4 = 1250
    
    def luw_construct(k, g):
        if g % 2 == 1:
            d = (g - 1) // 2
        else:
            d = g // 2
        
        q = k
        
        if d == 4 and q == 5:
            # Specific construction for (5,9)
            # Vertex: (side, x0, x1, x2, x3) where side in {0,1}, xi in Z_5
            
            def vertex_id(side, x0, x1, x2, x3):
                return side * 625 + x0 * 125 + x1 * 25 + x2 * 5 + x3
            
            n = 2 * 625
            adj = [set() for _ in range(n)]
            
            for side in range(2):
                for x0 in range(5):
                    for x1 in range(5):
                        for x2 in range(5):
                            for x3 in range(5):
                                u = vertex_id(side, x0, x1, x2, x3)
                                # Neighbors: (1-side, y0, y1, y2, y3)
                                # y1 = t (free)
                                # y2 = x2 + t*x1 (mod 5)
                                # y3 = x3 + t*x2 + t^2*x1 (mod 5)
                                # y0 = x0 + t*x3 + t^2*x2 + t^3*x1 (mod 5)
                                for t in range(5):
                                    y1 = t
                                    y2 = (x2 + t * x1) % 5
                                    y3 = (x3 + t * x2 + t * t * x1) % 5
                                    y0 = (x0 + t * x3 + t * t * x2 + t * t * t * x1) % 5
                                    v = vertex_id(1 - side, y0, y1, y2, y3)
                                    adj[u].add(v)
            
            adj_list = [list(s) for s in adj]
            edges = []
            for i in range(n):
                for j in adj_list[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges, adj_list
        
        # Generic fallback for other parameters
        # This is a simplified version that may not always work
        d_eff = d
        q_eff = q
        n = 2 * (q_eff ** d_eff)
        
        # Build adjacency using a simple rule
        adj = [set() for _ in range(n)]
        
        for side in range(2):
            for x_tuple in range(q_eff ** d_eff):
                u = side * (q_eff ** d_eff) + x_tuple
                x = [0] * d_eff
                temp = x_tuple
                for i in range(d_eff - 1, -1, -1):
                    x[i] = temp % q_eff
                    temp //= q_eff
                
                for t in range(q_eff):
                    y = [0] * d_eff
                    y[0] = t
                    for j in range(1, d_eff):
                        val = x[j]
                        for i in range(1, j + 1):
                            val += pow(t, i, q_eff) * x[j - i]
                        y[j] = val % q_eff
                    
                    y_tuple = 0
                    for i in range(d_eff):
                        y_tuple = y_tuple * q_eff + y[i]
                    v = (1 - side) * (q_eff ** d_eff) + y_tuple
                    adj[u].add(v)
        
        adj_list = [list(s) for s in adj]
        edges = []
        for i in range(n):
            for j in adj_list[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges, adj_list

    # Try LUW construction
    best_result = None
    best_n = float('inf')
    
    n_luw, edges_luw, adj_luw = luw_construct(k, g)
    result = validate(adj_luw, n_luw, k, g)
    if result is not None:
        best_n = n_luw
        best_result = result
    
    # If LUW works, return it
    if best_result is not None:
        return best_result
    
    # Fallback: try random regular graphs
    def random_k_regular(n, k):
        if n * k % 2 != 0:
            return None
        
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        
        random.shuffle(half_edges)
        
        adj = [set() for _ in range(n)]
        used = [False] * len(half_edges)
        
        i = 0
        while i < len(half_edges):
            if used[i]:
                i += 1
                continue
            u = half_edges[i]
            found = False
            for j in range(i + 1, len(half_edges)):
                if not used[j] and half_edges[j] != u:
                    v = half_edges[j]
                    if v not in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                        used[i] = True
                        used[j] = True
                        found = True
                        break
            if not found:
                return None
            i += 1
        
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        return [list(s) for s in adj]
    
    # Moore bound for (5,9) = 426
    # Try random graphs from 430 to 600
    for n_try in range(430, 600, 5):
        if remaining() < 20:
            break
        if n_try * k % 2 != 0:
            continue
        
        for attempt in range(3):
            if remaining() < 20:
                break
            adj = random_k_regular(n_try, k)
            if adj is None:
                continue
            if not is_connected(adj, n_try):
                continue
            if check_girth_bfs(adj, n_try, g):
                edges = []
                for i in range(n_try):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n = n_try
                best_result = (n_try, edges)
                break
    
    if best_result is not None:
        return best_result
    
    # Last resort: return LUW even if validation failed (shouldn't happen)
    return (n_luw, edges_luw)

# EVOLVE-BLOCK-END
