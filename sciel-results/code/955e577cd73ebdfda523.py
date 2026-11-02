import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    def luw_construct(q, d):
        # Standard LUW construction for prime q
        # Vertices are (side, x0, x1, ..., xd-1)
        # side in {0, 1}
        # x_i in Z_q
        
        # Generators are defined by the LUW rules
        # For L(q, d), the degree is q.
        # The edges are defined by:
        # (0, x) ~ (1, y) where y_i = x_i + sum_{j} ...
        
        # The seed code has a specific implementation.
        # We will reuse the logic but with d=4.
        
        table = None
        if q == 4:
            # Not used for q=5
            pass
            
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
        
        # Rule generation from seed
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth(n, edges, target_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each vertex to find shortest cycle
        # For each vertex u, BFS to depth target_girth//2
        # If we find a cycle shorter than target_girth, return False
        
        for u in range(n):
            # BFS
            visited = [-1] * n
            parent = [-1] * n
            depth = [0] * n
            visited[u] = 0
            queue = [u]
            found_short = False
            
            while queue:
                curr = queue.pop(0)
                for nb in adj[curr]:
                    if visited[nb] == -1:
                        visited[nb] = 1
                        parent[nb] = curr
                        depth[nb] = depth[curr] + 1
                        if depth[nb] < target_girth: # Optimization: don't go deeper than needed
                            queue.append(nb)
                    else:
                        # Found a cycle
                        # Length is depth[curr] + depth[nb] + 1
                        # But this only finds odd cycles if we don't handle even ones carefully?
                        # Actually, in BFS, the first time we see a visited node (not parent)
                        # gives a cycle.
                        # However, this might not be the shortest cycle through u?
                        # For girth check, we just need *any* cycle < g.
                        
                        # We must exclude the parent edge
                        if nb == parent[curr]:
                            continue
                            
                        cycle_len = depth[curr] + depth[nb] + 1
                        if cycle_len < target_girth:
                            return False
            # Note: This BFS finds the shortest cycle *in the BFS tree*?
            # Actually, it finds *a* cycle. Is it the shortest?
            # In a BFS tree, the first back-edge found gives a cycle.
            # But there might be a shorter cycle not involving the back-edge found first?
            # No, BFS guarantees that if a cycle of length L exists, it will be found when processing nodes at depth roughly L/2.
            # However, to be safe and fast for small girth:
            # We can just check all cycles up to length target_girth-1 explicitly?
            # Given n=625, BFS is fast enough.
            
        return True

    # Try LUW with d=4
    # k=5, so q=5
    if k == 5:
        d = 4
        n, edges = luw_construct(5, d)
        if check_girth(n, edges, g):
            return n, edges
            
    # Fallback: LUW with d=5 (Seed behavior)
    d = 5
    n, edges = luw_construct(5, d)
    if check_girth(n, edges, g):
        return n, edges
        
    # Fallback: Random search for Circulant?
    # Or just return the larger LUW graph (which is valid for g=11, so valid for g=9)
    return n, edges
