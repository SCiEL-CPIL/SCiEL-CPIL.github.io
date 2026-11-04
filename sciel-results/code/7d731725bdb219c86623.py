import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k != 3:
        # Fallback for other k, though task is 3-regular
        # For now, return a simple cycle or similar if k != 3?
        # The task says "3-regular". I will assume k=3.
        pass

    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    def build_luw(d):
        rule = {}
        if d >= 2: rule[2] = (1, 1)
        if d >= 3: rule[3] = (2, 1)
        if d >= 4: rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
        def get_neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y))
                
        start_vertex = (0, tuple([0]*d))
        visited = {start_vertex: 0}
        order = [start_vertex]
        edges = []
        
        for vertex in order:
            idx = visited[vertex]
            for neighbor in get_neighbours(vertex):
                if neighbor not in visited:
                    visited[neighbor] = len(order)
                    order.append(neighbor)
                if vertex[0] == 0:
                    edges.append((idx, visited[neighbor]))
                    
        n = len(order)
        return n, edges

    def check_girth(n, edges, g):
        R = g // 2 - 1
        if R < 0:
            return True
        expected_size = 1 + 3 * (2**R - 1) if R > 0 else 1
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for v in range(n):
            dist = [-1] * n
            dist[v] = 0
            q_bfs = deque([v])
            count = 0
            while q_bfs:
                curr = q_bfs.popleft()
                count += 1
                if dist[curr] == R:
                    continue
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        q_bfs.append(nei)
            if count < expected_size:
                return False
        return True

    best_n = float('inf')
    best_edges = None
    
    # Try d from 3 up to g-5
    # g=16 -> d up to 11
    for d in range(3, g - 2): 
        if time.time() > deadline:
            break
            
        n, edges = build_luw(d)
        
        # Quick check: if n is already larger than best, skip girth check?
        # But we want the smallest n, so we iterate d upwards.
        # The first valid one is the best.
        
        if check_girth(n, edges, g):
            best_n = n
            best_edges = edges
            break
            
    if best_n == float('inf'):
        # Fallback to theoretical bound
        d = g - 5 if (g - 5) % 2 else g - 4
        best_n, best_edges = build_luw(d)
        
    return best_n, best_edges
