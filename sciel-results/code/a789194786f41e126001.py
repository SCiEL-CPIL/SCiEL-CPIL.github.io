import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Moore bound calculation
    # For odd girth g = 2r + 1
    if g % 2 == 1:
        r = (g - 1) // 2
        # M(k, g) = 1 + k * sum_{i=0}^{r-1} (k-1)^i
        # Sum is geometric series: ( (k-1)^r - 1 ) / (k-2) if k > 2
        if k == 2:
            moore = 2 * r + 1
        else:
            moore = 1 + k * (( (k-1)**r - 1 ) // (k-2))
    else:
        # Even girth
        r = g // 2
        # M(k, g) = 2 * sum_{i=0}^{r-1} (k-1)^i
        if k == 2:
            moore = 2 * r
        else:
            moore = 2 * ( ( (k-1)**r - 1 ) // (k-2) )
            
    # Heuristic start: Random graphs need roughly k^(g/2 - 1) vertices
    # For k=5, g=9, r=4, 5^4 = 625.
    # Known best is 1116.
    # Let's start scanning from max(moore, 800) up to 1500
    
    n_start = max(moore, 800)
    n_end = 1500
    
    # If k is small or g is small, adjust range
    if k == 5 and g == 9:
        n_start = 1000
        n_end = 1200
        
    def is_valid_graph(n, k, g):
        # Generate random k-regular graph
        # Use configuration model
        for _ in range(100): # Try 100 times to get a simple one
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            simple = True
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    simple = False
                    break
                if v in adj[u]:
                    simple = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if not simple:
                continue
            
            # Check connectivity
            visited = [False] * n
            stack = [0]
            visited[0] = True
            count = 0
            while stack:
                u = stack.pop()
                count += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            if count != n:
                continue
            
            # Check girth
            # If girth < g, invalid
            # We check for cycles of length < g
            # For each vertex, BFS up to depth g//2
            # If we find a back edge that creates a cycle of length < g, invalid
            # Note: For girth g, we need no cycles of length 3, 4, ..., g-1.
            # BFS from each vertex:
            # dist[u] = distance from source
            # parent[u] = parent in BFS tree
            # If we see an edge (u, v) where v is visited:
            #   if u == parent[v] or v == parent[u], skip (tree edge)
            #   else, cycle length = dist[u] + dist[v] + 1
            #   if cycle length < g, invalid
            
            h = g // 2
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = [s]
                # BFS
                while queue:
                    # Use list as queue, but for small depth, stack is fine too. 
                    # Let's use list and pop(0) is slow, so use collections.deque or just list with index
                    # For simplicity and speed in Python, list with index or stack (DFS) works.
                    # Girth check with DFS:
                    # If we find a back edge, check cycle length.
                    # DFS is better for girth?
                    # Let's stick to BFS for clarity, but optimize.
                    pass
                
                # Optimized Girth Check using BFS from each node
                # Since g is small (9), depth is 4.
                # We can just do a bounded BFS.
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = [s]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    if dist[u] >= h:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            # Found a visited node
                            # Check if it's the parent (tree edge)
                            if v == parent[u]:
                                continue
                            # Check if u is parent of v?
                            # In BFS, if dist[v] == dist[u] - 1, then v is parent of u?
                            # Actually, if v is visited, and not parent, it's a cross/back edge.
                            # Cycle length = dist[u] + dist[v] + 1
                            # But this is only valid if the paths are disjoint except for endpoints?
                            # In a tree, yes. In BFS tree, the paths are unique.
                            # So if we encounter a visited node v (not parent),
                            # the cycle length is dist[u] + dist[v] + 1.
                            # However, this might overcount if the paths share nodes?
                            # No, BFS tree paths are simple.
                            # But wait, if dist[v] < dist[u], and v is not parent,
                            # then we have a cycle.
                            # We need to ensure the cycle length < g.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
            # If no short cycles found, valid
            return True
        
        return False

    best_n = None
    best_edges = None
    
    # Search for smallest n
    for n in range(n_start, n_end + 1):
        if time.time() > deadline:
            break
        
        # Try to find a valid graph for this n
        # Number of attempts depends on remaining time
        remaining = deadline - time.time()
        attempts = int(remaining * 5) # Tune this
        if attempts < 10:
            attempts = 10
            
        for _ in range(attempts):
            if time.time() > deadline:
                break
            # Generate and check
            # We need to return edges, so modify is_valid to return edges or None
            # Let's refactor: generate, check, return edges if valid
            
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            simple = True
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    simple = False
                    break
                if v in adj[u]:
                    simple = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if not simple:
                continue
            
            # Connectivity
            visited = [False] * n
            stack = [0]
            visited[0] = True
            count = 0
            while stack:
                u = stack.pop()
                count += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            if count != n:
                continue
            
            # Girth Check
            valid = True
            h = g // 2
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = [s]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    if dist[u] >= h:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if v == parent[u]:
                                continue
                            # Check if u is parent of v?
                            # If dist[v] == dist[u] + 1, then v is child, u is parent.
                            # If dist[v] == dist[u] - 1, then v is parent, u is child.
                            # If dist[v] == dist[u], cross edge.
                            # In any case, if not parent, it's a cycle.
                            # Cycle length = dist[u] + dist[v] + 1
                            # Is this always correct?
                            # Yes, for BFS tree.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Build edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # If no graph found, fallback to a known valid construction or larger n
    # Use L-U-W from seed if time permits?
    # Or just try larger n
    for n in range(n_end + 1, n_end + 200):
        if time.time() > deadline:
            break
        # Quick check for one valid graph
        for _ in range(10):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            simple = True
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v or v in adj[u]:
                    simple = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not simple: continue
            # Connectivity
            visited = [False] * n
            stack = [0]
            visited[0] = True
            count = 0
            while stack:
                u = stack.pop()
                count += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            if count != n: continue
            # Girth
            valid = True
            h = g // 2
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = [s]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    if dist[u] >= h: continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if v == parent[u]: continue
                            if dist[u] + dist[v] + 1 < g:
                                valid = False
                                break
                    if not valid: break
                if not valid: break
            if valid:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Absolute fallback: Return a trivial valid graph if possible?
    # No, must be k-regular.
    # If we run out of time, we must return something.
    # The seed program returns a valid graph. We should ensure we return a valid graph.
    # If random search fails, we can use the L-U-W graph from the seed, but it's slow to generate?
    # Actually, L-U-W generation is fast.
    # Let's just return the result of the seed logic if we haven't found anything.
    # But we are rewriting.
    # Let's just try a very large n that is guaranteed to work?
    # For k=5, g=9, n=2000 will definitely have a girth-9 graph.
    # Let's try n = 2000 once.
    n = 2000
    for _ in range(5):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        simple = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v or v in adj[u]:
                simple = False
                break
            adj[u].append(v)
            adj[v].append(u)
        if not simple: continue
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 0
        while stack:
            u = stack.pop()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if count != n: continue
        valid = True
        h = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= h: continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v == parent[u]: continue
                        if dist[u] + dist[v] + 1 < g:
                            valid = False
                            break
                if not valid: break
            if not valid: break
        if valid:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    # If all else fails, return a dummy valid graph?
    # This should not happen.
    return 0, []
