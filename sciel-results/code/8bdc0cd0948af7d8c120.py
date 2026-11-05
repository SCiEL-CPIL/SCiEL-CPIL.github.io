import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # Local Search for smaller graph
    def local_search(k, g, n_target, time_limit=140):
        start_time = time.time()
        # Generate random k-regular graph on n_target vertices
        # Using configuration model or simple swaps
        # For k=4, n=320
        # We'll use a simple random regular graph generator
        def generate_random_regular_graph(n, k):
            # Stirling's cycle method or similar
            # For simplicity, use a random permutation based method
            # Or just random edges and fix degrees
            # Better: Use a known algorithm for random regular graphs
            # Here we use a simple "random matching" for k/2 if k even? No, k=4.
            # We can use the "random swap" method starting from a fixed structure.
            # Start with a cycle of length n (2-regular) and add a random 2-regular graph (union of cycles).
            # Union of two 2-regular graphs is 4-regular.
            # 2-regular graphs are unions of cycles.
            # So we pick two random permutations (cycles) and combine them.
            
            # Generate two random permutations (as lists of next nodes)
            # Perm 1
            p1 = list(range(n))
            random.shuffle(p1)
            # Convert to adjacency
            adj1 = {}
            for i in range(n):
                u = p1[i]
                v = p1[(i+1)%n]
                adj1[u] = [v, p1[(i-1)%n]]
            
            # Perm 2
            p2 = list(range(n))
            random.shuffle(p2)
            adj2 = {}
            for i in range(n):
                u = p2[i]
                v = p2[(i+1)%n]
                adj2[u] = [v, p2[(i-1)%n]]
            
            # Combine
            adj = {}
            for i in range(n):
                adj[i] = adj1[i] + adj2[i]
                # Remove duplicates if any (should not happen if perms are disjoint? No, they can share edges)
                # Ensure no multi-edges
                adj[i] = list(set(adj[i]))
                if len(adj[i]) < k:
                    # This can happen if edges overlap. 
                    # For k=4, if we use two disjoint permutations, it's fine.
                    # But random perms might create multi-edges.
                    # We'll just accept it and fix later or restart.
                    pass
            
            # Check if valid
            valid = True
            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
                if len(set(adj[i])) != k:
                    valid = False
                    break
                for j in adj[i]:
                    if i in adj[j]:
                        if adj[j].count(i) > 1:
                            valid = False
                    else:
                        valid = False
            if not valid:
                return None, None
            
            # Build edge list
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return adj, edges

        best_n = float('inf')
        best_edges = None
        
        # Try a few restarts
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            adj, edges = generate_random_regular_graph(n_target, k)
            if adj is None:
                continue
            
            # Check girth
            def get_girth(adj, n, k, max_depth=5):
                # BFS from each vertex up to depth max_depth
                # If we find a cycle of length < g, return it
                # We want to know the minimum cycle length
                min_girth = g + 1 # Assume valid
                for start in range(n):
                    # BFS
                    visited = [-1] * n
                    parent = [-1] * n
                    depth = [0] * n
                    queue = deque()
                    queue.append(start)
                    visited[start] = start
                    while queue:
                        u = queue.popleft()
                        if depth[u] > max_depth:
                            continue
                        for v in adj[u]:
                            if visited[v] == -1:
                                visited[v] = u
                                parent[v] = u
                                depth[v] = depth[u] + 1
                                queue.append(v)
                            elif v != parent[u] and visited[v] != u:
                                # Found a cycle
                                # Calculate length
                                l1 = depth[u]
                                l2 = depth[v]
                                # Trace back to LCA
                                # Simplified: just check if depth sum + 1 is small
                                # This is an approximation. 
                                # For exact girth, we need proper LCA.
                                # Given time, we'll just use a heuristic: 
                                # If we find a back-edge, cycle length is at least 3.
                                # We'll just return a conservative girth estimate.
                                pass
                return min_girth # This is a stub, needs real impl

            # Real Girth Check (BFS from each vertex, stop at depth g//2)
            def check_girth(adj, n, g):
                for start in range(n):
                    visited = [-1] * n
                    parent = [-1] * n
                    depth = [0] * n
                    queue = deque()
                    queue.append(start)
                    visited[start] = start
                    while queue:
                        u = queue.popleft()
                        if depth[u] >= g // 2:
                            continue
                        for v in adj[u]:
                            if visited[v] == -1:
                                visited[v] = u
                                parent[v] = u
                                depth[v] = depth[u] + 1
                                queue.append(v)
                            elif v != parent[u]:
                                # Cycle found
                                # Length is depth[u] + depth[v] + 1
                                # If < g, return False
                                # But we need exact length? No, just < g.
                                # However, this detects ANY cycle. 
                                # We need to ensure NO cycle < g.
                                # So if we find a cycle of length L < g, return False.
                                # We need to calculate L.
                                # Trace back from u and v to LCA.
                                # For speed, we'll just return False if any back-edge is found 
                                # within depth g//2? 
                                # No, a back-edge within depth 4 implies cycle <= 9.
                                # So if we find a back-edge at depth <= 4, girth < 10.
                                return False
                return True

            # Local Search Loop
            current_girth_valid = check_girth(adj, n_target, g)
            if current_girth_valid:
                return n_target, edges
            
            # Try to improve
            for step in range(10000):
                if time.time() - start_time > time_limit:
                    break
                
                # Pick two random edges
                # We need to pick (u,v) and (w,x)
                # and swap to (u,w) and (v,x)
                # Ensure no loops/multi-edges
                
                # Randomly pick u, v
                u = random.randint(0, n_target - 1)
                v = random.choice(adj[u])
                
                # Randomly pick w, x
                w = random.randint(0, n_target - 1)
                x = random.choice(adj[w])
                
                # Check distinctness
                if u == w or u == x or v == w or v == x:
                    continue
                
                # Check if (u,w) and (v,x) are valid
                if w in adj[u] or x in adj[v]:
                    continue
                if u in adj[w] or v in adj[x]:
                    continue
                
                # Perform swap
                adj[u].remove(v)
                adj[w].remove(x)
                adj[u].append(w)
                adj[w].append(u)
                adj[v].append(x)
                adj[x].append(v)
                
                # Check girth
                if check_girth(adj, n_target, g):
                    # Build edges
                    edges = []
                    for i in range(n_target):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n_target, edges

        return None, None

    # Try local search
    # Target n=320
    n_ls, edges_ls = local_search(k, g, 320, 140)
    if n_ls is not None:
        return n_ls, edges_ls
    
    # Fallback to LUW
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
