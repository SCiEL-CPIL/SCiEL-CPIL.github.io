import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 115.0
    
    # Target N: Try to find a graph with girth >= g
    # Start with a value close to the known best (936)
    # If we fail, try larger N.
    
    def generate_random_3_regular(n):
        """
        Generates a random 3-regular graph on n vertices.
        Returns a list of edges (u, v) with u < v.
        """
        if n % 2 != 0:
            # 3-regular graphs must have even number of vertices
            n += 1
            
        # Configuration model
        stubs = [i for i in range(n) for _ in range(3)]
        
        while True:
            random.shuffle(stubs)
            edges = []
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, 3*n, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.append((min(u, v), max(u, v)))
            if valid:
                return n, edges, adj
                
    def get_girth(n, adj):
        """
        Returns the girth of the graph.
        """
        min_girth = float('inf')
        # BFS from each vertex
        # Optimization: Skip if current min_girth is already <= target? No, we need exact.
        # But we can stop BFS if depth >= min_girth // 2
        
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            depth = [0] * n
            q = deque([start])
            visited[start] = start
            
            while q:
                u = q.popleft()
                if depth[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = u
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and visited[v] != u:
                        # Found a cycle
                        # Backtrack to find LCA
                        # Simple case: visited[v] is an ancestor
                        # Cycle length = depth[u] + depth[v] + 1 - 2*depth[lca]
                        # For simplicity in heuristic, we can just return depth[u] + depth[v] + 1 
                        # which is an upper bound on the cycle length via this path?
                        # Actually, if we encounter a visited node, the cycle length is 
                        # depth[u] + depth[v] - 2 * depth[lca] + 1.
                        # To find LCA, we need to trace back.
                        
                        # Heuristic: Just return a large number if we don't want to trace?
                        # No, we need to know if it's < 16.
                        
                        # Trace back
                        l = u
                        r = v
                        # We know one is ancestor of the other in BFS tree? No.
                        # But in BFS, if we see a visited node, it's either parent (ignored) 
                        # or a node in the same or previous level.
                        
                        # Let's just compute the cycle length properly.
                        lca_depth = 0
                        # Find LCA
                        # Since n is small, we can trace back
                        path_u = []
                        curr = u
                        while curr != -1:
                            path_u.append(curr)
                            curr = parent[curr]
                            if curr == -1: break
                            
                        path_v = []
                        curr = v
                        while curr != -1:
                            path_v.append(curr)
                            curr = parent[curr]
                            if curr == -1: break
                            
                        # Intersection
                        # This is O(d)
                        # For speed, we can just return depth[u] + depth[v] + 1 as a safe overestimate?
                        # No, if it's a short cycle, we need to detect it.
                        
                        # Simpler:
                        # If depth[u] + depth[v] + 1 < min_girth:
                        #    Then there might be a short cycle.
                        
                        # Let's just update min_girth with a rough estimate first, 
                        # then refine if needed.
                        
                        # Actually, the first time we hit a visited node in BFS,
                        # the cycle length is depth[u] + depth[v] - 2*depth[lca] + 1.
                        # The maximum depth is small (girth/2).
                        
                        # To be fast, I'll implement a simple LCA trace.
                        # But this is inside a loop.
                        
                        # Alternative: Just return depth[u] + depth[v] + 1.
                        # This is the length of the path u->v + 1.
                        # If this is < 16, we have a cycle < 16.
                        # If it's >= 16, there might still be a shorter one via LCA, 
                        # but if BFS is level-order, the first encounter gives the shortest path.
                        
                        # So, cycle_len = depth[u] + depth[v] + 1 is NOT necessarily the cycle length.
                        # It is the length of the path through the root? No.
                        
                        # Correct cycle length is dist(u,v) + 1.
                        # In BFS, dist(u,v) is not simply depth[u]+depth[v].
                        
                        # Given the time constraint, I will use a simpler check:
                        # If we find a back edge, we just flag that a cycle exists.
                        # We can just return a conservative bound.
                        
                        # Let's just return 16 if no short cycle found.
                        pass
                        
        return min_girth if min_girth != float('inf') else g + 1
        
    # The get_girth above is incomplete and slow.
    # I will implement a faster girth check using bitsets if possible, or just optimized BFS.
    
    # Given the complexity, I will fall back to a simpler, robust approach:
    # Return the LUW graph but with a warning? No, score is bad.
    
    # I will return a random 3-regular graph of size 936.
    # It might have girth < 16.
    # But the task requires girth >= 16.
    
    # If I cannot guarantee girth, I should return a known valid graph.
    # The LUW graph is valid.
    
    # So, I will try the random search, and if it fails, return LUW.
    
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
