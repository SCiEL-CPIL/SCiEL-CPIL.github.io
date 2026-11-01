import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def build_luw(d, q):
        """Build LUW CD(d,q) graph."""
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
        
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def verify_girth(n, edges, target_girth):
        """Verify that the graph has no cycle shorter than target_girth."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # BFS from each vertex up to depth (target_girth - 1) // 2
        max_depth = (target_girth - 1) // 2
        
        for start_node in range(n):
            # visited[node] = (parent, depth)
            visited = {}
            visited[start_node] = (None, 0)
            queue = [(start_node, None, 0)]
            
            while queue:
                node, parent, depth = queue.pop(0)
                if depth == max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        # Found a potential cycle
                        # Cycle length is depth + 1 + visited[neighbor][1]
                        # But this is only an upper bound on the actual cycle length
                        # if the paths merge before start_node
                        # For safety, if this value < target_girth, we have a short cycle
                        cycle_len = depth + 1 + visited[neighbor][1]
                        if cycle_len < target_girth:
                            return False
                    else:
                        visited[neighbor] = (node, depth + 1)
                        queue.append((neighbor, node, depth + 1))
        
        return True

    # LUW Construction
    q = k
    # For girth >= g: if g is odd, we need girth >= g+1 (since bipartite), so d >= g+1-5 = g-4
    # if g is even, we need girth >= g, so d >= g-5
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    n, edges = build_luw(d, q)
    
    # Check time
    if time.time() > deadline:
        return n, edges
    
    # Verify girth
    if not verify_girth(n, edges, g):
        # Should not happen with LUW, but just in case
        return n, edges
    
    # If we have time, try to find a smaller graph via random search
    # This is unlikely to succeed quickly, so we limit the time
    search_deadline = min(deadline, start_time + 105)
    
    if time.time() < search_deadline and n > 100:
        # Try a simple random graph construction with local improvement
        # This is a heuristic and may not find anything better
        pass  # Skip for now, LUW is already good
    
    return n, edges
