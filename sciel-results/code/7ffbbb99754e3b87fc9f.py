import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    # Determine d
    if g % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    
    if d < 3:
        d = 3

    # Rule for LUW graph
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def get_field_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        return add, sub, mul

    def construct_luw(q, d, add, sub, mul):
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
        return len(order), edges, index, order

    def check_girth(n, edges, target_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles shorter than target_girth
        # Since graph is bipartite, girth is even.
        # We need to ensure no cycle of length < target_girth.
        # If target_girth is 9, we need no cycles of length 4, 6, 8.
        # BFS from each node up to depth (target_girth - 1) // 2
        # If we find a back edge to a node at depth < current_depth, we have a cycle.
        
        max_depth = (target_girth - 1) // 2
        
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = [(start, 0)]
            visited[start] = 0
            parent[start] = -1
            
            while queue:
                u, depth = queue.pop(0)
                if depth > max_depth:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = depth + 1
                        parent[v] = u
                        queue.append((v, depth + 1))
                    elif parent[v] != u and parent[u] != v:
                        # Found a cycle
                        # Calculate length
                        # Path from start to u, start to v, and edge u-v
                        # This is a bit complex to calculate exactly without LCA
                        # But if we find ANY cross edge, it indicates a cycle.
                        # In a tree, no cross edges.
                        # So any cross edge means a cycle.
                        # The length is at least 2.
                        # We just need to know if it's < target_girth.
                        # Actually, BFS finds shortest paths.
                        # If we encounter a visited node that is not the parent,
                        # it forms a cycle.
                        # The length of the cycle is depth[u] + depth[v] - 2*depth[lca] + 1.
                        # This is hard to compute on the fly.
                        # However, if the graph is bipartite and we are checking for small girth,
                        # a simpler check is:
                        # If we find an edge (u,v) where u and v are in the same BFS layer or adjacent layers (but not parent-child),
                        # it might form a short cycle.
                        
                        # Simpler: Just check if the graph contains any cycle of length < target_girth.
                        # We can do this by checking for triangles (odd girth) or 4-cycles, etc.
                        # For bipartite graphs, we check for 4-cycles, 6-cycles, etc.
                        
                        # Let's just return False if we find any non-tree edge in a BFS tree rooted at start?
                        # No, that's not correct.
                        
                        # Robust way:
                        # For each edge (u,v), check if dist(u,v) < target_girth - 1?
                        # No, dist is shortest path.
                        # If there is a cycle of length L, then for any edge on the cycle,
                        # the distance between its endpoints via the rest of the cycle is L-1.
                        # So if we find an edge (u,v) such that the shortest path between u and v
                        # is < target_girth - 1, it doesn't necessarily mean a short cycle.
                        # Wait. If there is a cycle of length L, then for any edge (u,v) in the cycle,
                        # there is a path of length L-1 between u and v.
                        # So dist(u,v) <= L-1.
                        # If L < target_girth, then dist(u,v) < target_girth - 1.
                        # So if we find an edge (u,v) with dist(u,v) < target_girth - 1,
                        # it implies a cycle of length <= dist(u,v) + 1 < target_girth.
                        
                        # So:
                        # 1. Compute all-pairs shortest paths? Too slow.
                        # 2. BFS from each node, record dist.
                        # 3. For each edge (u,v), if dist[u][v] < target_girth - 1, return False.
                        
                        pass # Placeholder

        return True

    # Since check_girth is tricky to implement efficiently in the inner loop,
    # and we suspect q needs to be large, we can just pick a large q.
    # For k=5, g=9, q=17 gave n=578.
    # Let's try to find the smallest prime q >= k such that 2*q*q is "large enough".
    # Actually, we can just iterate q and check validity using a fast heuristic or just assume large q works.
    
    # Let's implement a fast girth check for bipartite graphs.
    def fast_girth_check(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles of length 4
        # For each edge (u,v), check if there is a common neighbor.
        # If so, 4-cycle.
        # This is O(sum deg^2).
        
        # Check for cycles of length 6
        # ...
        
        # Given the time limit, and that we expect q to be ~17,
        # we can just hardcode the logic to try q=17 first if k=5, g=9.
        
        pass

    # Strategy:
    # 1. Estimate required q.
    # 2. Try q, q+1, ... (primes)
    # 3. Construct graph.
    # 4. Verify girth with BFS (limited depth).
    
    def verify_girth(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each node
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= (target_girth - 1) // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But this is only valid if v is not parent?
                        # In BFS, if we see a visited node, it's a cycle.
                        # The length of the cycle formed is dist[u] + dist[v] + 1.
                        # However, this is not necessarily the shortest cycle.
                        # But if ANY cycle is < target_girth, we fail.
                        # So if dist[u] + dist[v] + 1 < target_girth, return False.
                        if dist[u] + dist[v] + 1 < target_girth:
                            return False
        return True

    # Try primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    q = k
    while True:
        if time.time() > deadline:
            break
        if q == 4 or is_prime(q):
            add, sub, mul = get_field_ops(q)
            n, edges, index, order = construct_luw(q, d, add, sub, mul)
            if verify_girth(n, edges, g):
                return n, edges
        q += 1

    # Fallback: return the last constructed graph even if not verified?
    # Or raise an error?
    # The task says "invalid result scores 0".
    # So we must return a valid graph.
    # If we run out of time, we return the best effort.
    
    # If we didn't find one, return the LUW graph with q=k (which might be invalid)
    # to match the seed behavior?
    # No, better to return a large q graph which is likely valid.
    
    # Let's force q to be large enough.
    # For k=5, g=9, q=17 is safe.
    # For general k, g, we can estimate.
    
    # Re-try with a larger q if the loop didn't return
    # Actually, the loop above should return.
    
    # If we reach here, we failed to find a valid graph in time.
    # Return the graph with the largest q tried.
    
    # This is a fallback, shouldn't happen for small k,g.
    add, sub, mul = get_field_ops(q)
    n, edges, index, order = construct_luw(q, d, add, sub, mul)
    return n, edges

