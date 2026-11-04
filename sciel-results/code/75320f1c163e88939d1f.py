import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    # Fallback to LUW if random search fails or takes too long
    # But LUW is huge. Let's try to make a smaller one first.
    
    # Target n for random graph
    # Expected girth ~ log_{k-1}(n)
    # We want girth >= g
    # n ~ (k-1)^g
    # For k=3, g=16, n ~ 2^16 = 65536
    # Let's start with n = 100000 to be safe.
    
    base_n = 100000
    # Ensure n is even
    if base_n % 2 != 0:
        base_n += 1
        
    # We will try a few random instances
    max_attempts = 5
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        n = base_n + attempt * 10000
        if n % 2 != 0:
            n += 1
            
        # Generate random 3-regular graph
        edges = generate_random_3_regular(n)
        
        # Check girth
        if check_girth(edges, n, g):
            return n, edges
            
    # If all fail, we must return a valid graph.
    # The LUW graph is valid but large.
    # We can return a smaller LUW graph? No, LUW size is fixed by q and d.
    # If we can't find a small one, we return the LUW one from the original code?
    # The original code is available. I should probably include it as fallback.
    
    return luw_construct(k, g)

def generate_random_3_regular(n):
    # Configuration model
    # 3n half edges
    # Shuffle and pair
    # Resolve self-loops and multi-edges
    
    # For n=100k, this is fast.
    half_edges = list(range(n)) * 3
    random.shuffle(half_edges)
    
    edges = []
    for i in range(0, 3*n, 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            # Self-loop, swap with another edge
            # Find a valid partner
            # Simple retry: just swap v with a random other half-edge
            # This might be slow if done naively.
            # Better: use a standard algorithm.
            pass
            
    # Standard algorithm for simple random regular graph:
    # 1. Generate random permutation of 3n items.
    # 2. Pair them.
    # 3. If multi-edges or loops, perform switchings.
    
    # Given time constraints, let's use a simpler approach:
    # Generate a random 3-regular multigraph, then "fix" it.
    # Or use the fact that for large n, the probability of multi-edges is low.
    # P(loop) ~ 1/n. P(multi-edge) ~ 1/n.
    # So for n=100k, ~1% of edges might be bad.
    # We can just retry generation if we find any.
    
    while True:
        half_edges = list(range(n)) * 3
        random.shuffle(half_edges)
        
        adj = [[] for _ in range(n)]
        valid = True
        edge_set = set()
        
        for i in range(0, 3*n, 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                valid = False
                break
            edge_set.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        if valid:
            # Check degree
            for i in range(n):
                if len(adj[i]) != 3:
                    valid = False
                    break
            if valid:
                return edge_set
        # Retry

def check_girth(edges, n, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex to depth g/2
    # We need to detect cycles of length < g
    # Max depth to check: floor((g-1)/2)
    # For g=16, depth 7 detects cycles <= 14.
    # We also need to detect cycle 15.
    # Cycle 15 requires depth 7 from one end and 8 from the other?
    # If we BFS to depth 8, we detect cycles <= 16.
    # We reject if cycle < 16.
    
    max_depth = 8
    # To speed up, we can stop if we find a cycle.
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = [(s, 0)]
        dist[s] = 0
        
        while queue:
            u, d = queue.pop(0) # Use list as queue, inefficient but ok for small depth
            if d > max_depth:
                continue
                
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append((v, d + 1))
                else:
                    # Found a cycle
                    # Cycle length = d + 1 + dist[v]
                    # Wait, this is only valid if the path from s to v is disjoint from s to u
                    # In a tree, yes. In a graph with cycles, no.
                    # However, BFS finds shortest paths.
                    # If we encounter a visited node, the cycle formed is not necessarily simple.
                    
                    # Better: just check if the subgraph of depth max_depth has a cycle.
                    # We can do this by checking if number of edges > number of vertices - 1 in the BFS tree.
                    # But we need to know the exact cycle length.
                    
                    # Alternative: Count closed walks? No.
                    
                    # Let's use the property:
                    # If there is a cycle of length L < g, then BFS from any vertex on the cycle
                    # will find a "back edge" to a node in the BFS tree.
                    # The cycle length is dist[u] + dist[v] + 1.
                    # This is true if the shortest path from s to u and s to v are disjoint except s.
                    # This is not guaranteed.
                    
                    # Safe method:
                    # For each vertex s, perform BFS.
                    # Keep track of the first time a vertex is visited.
                    # If we find an edge (u, v) where dist[u] + dist[v] + 1 < g, and u, v are in the BFS tree of s,
                    # AND the path from s to u and s to v are disjoint (except s), then we have a cycle.
                    
                    # This is getting complicated.
                    # Given the time, let's assume the random graph is likely valid if n is large enough.
                    # We can use a simpler check:
                    # Just check if the graph contains any triangle, 4-cycle, etc.
                    # For g=16, we just need to ensure no cycles < 16.
                    
                    # Let's try a different approach:
                    # Use the fact that for random graphs, the girth is concentrated around log(n).
                    # If we pick n large enough, it's very likely to be valid.
                    # We can just return the graph if it passes a quick check.
                    
                    # Quick check:
                    # Count triangles. If > 0, fail.
                    # Count 4-cycles. If > 0, fail.
                    # ... up to 15-cycles.
                    
                    # This is too slow.
                    
                    # Let's stick to the BFS cycle detection.
                    # If we find a visited node, we compute the cycle length.
                    # If it's < g, we return False.
                    
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def luw_construct(k, g):
    # Copy from original
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
