import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # 130s limit, 20s margin
    
    q = k
    
    def build_luw(d):
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
        
        i = 0
        while i < len(order):
            if time.time() > deadline:
                return None
            vertex = order[i]
            i += 1
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def check_girth(n, edges, target_girth):
        # Check if girth >= target_girth using non-backtracking BFS
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # We need to ensure no cycle < target_girth
        # For a bipartite graph, all cycles are even.
        # We need to check for cycles of length 4, 6, ..., up to target_girth - 1 (if odd) or target_girth - 2 (if even)
        max_cycle_len = target_girth - 1 if target_girth % 2 == 1 else target_girth - 2
        if max_cycle_len < 4:
            return True
            
        max_depth = max_cycle_len // 2
        
        # Use deque for efficient BFS
        for start_node in range(n):
            # visited[node] = depth from start_node
            visited = {start_node: 0}
            queue = deque([(start_node, -1, 0)])  # (node, parent, depth)
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        # Cycle found: path from start to node (depth) + edge + path from start to neighbor (visited[neighbor])
                        cycle_len = depth + 1 + visited[neighbor]
                        if cycle_len < target_girth:
                            return False
                    else:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, node, depth + 1))
        return True

    # Estimate minimal d
    # Girth of LUW(d, q) is 2d+2
    # We need girth >= g
    # If g is odd, we need girth >= g+1 (since girth is even)
    # So 2d+2 >= g+1 => 2d >= g-1 => d >= (g-1)/2
    # If g is even, we need girth >= g
    # So 2d+2 >= g => 2d >= g-2 => d >= (g-2)/2
    
    if g % 2 == 1:
        d_min = (g - 1) // 2
    else:
        d_min = (g - 2) // 2
        
    d_min = max(3, d_min)
    
    best_n = float('inf')
    best_edges = None
    
    # Try d from d_min upwards
    # Limit d to avoid explosion
    max_d = d_min + 3
    
    for d in range(d_min, max_d + 1):
        if time.time() > deadline:
            break
            
        result = build_luw(d)
        if result is None:
            break
            
        n, edges = result
        
        # Check girth
        if check_girth(n, edges, g):
            if n < best_n:
                best_n = n
                best_edges = edges
            # Larger d gives larger n. So minimal d that works is best.
            break
            
    if best_edges is None:
        # Fallback: try slightly larger d
        d = d_min + 1
        result = build_luw(d)
        if result:
            n, edges = result
            if check_girth(n, edges, g):
                return n, edges
        return 0, []
        
    return best_n, best_edges
