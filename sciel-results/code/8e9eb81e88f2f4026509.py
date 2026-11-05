import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    # Moore bound
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
        
    # We want to find the smallest n.
    # Start from a reasonable estimate. For k=5, g=9, moore=106.
    # Random regular graphs of small n have small girth.
    # Let's try increasing n.
    
    best_n = None
    best_edges = None
    best_girth_found = 0
    
    # Helper to check girth >= g
    def has_girth_at_least_g(adj, n, g):
        # Check for cycles of length 3 to g-1
        # If none found, girth >= g
        # Use BFS on non-backtracking walks
        # To speed up, we can check all vertices.
        # Optimization: If we find a cycle of length L < g, return False.
        
        # For small n, this is fast.
        # We'll do BFS from each vertex.
        # State: (prev, curr)
        # dist[prev][curr]
        
        # To avoid allocating large arrays repeatedly, we can use a dictionary or 
        # a 2D list of lists, but numpy is faster.
        # However, for n=200, n*n=40000. 
        # We can use a single array of size n*n, reset it for each start vertex? 
        # That's n * (n*n) = n^3 operations just for resetting. 
        # Better: use a "visit_id" array.
        
        visit_id = np.zeros(n * n, dtype=np.int32)
        current_id = 1
        
        for start in range(n):
            # BFS
            # Queue of (prev, curr, depth)
            # Initialize
            queue = []
            for nb in adj[start]:
                idx = start * n + nb
                if visit_id[idx] != current_id:
                    visit_id[idx] = current_id
                    queue.append((start, nb, 1))
            
            # Process queue
            # We only care about depths up to g-1
            # If we reach (nb, start) at depth L, cycle length L.
            
            # To make it faster, we can stop BFS for this start if min depth > g-1?
            # No, we need to check if we can return to start.
            
            # Standard BFS
            while queue:
                new_queue = []
                for prev, curr, depth in queue:
                    if depth >= g - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        # Check if cycle back to start
                        if nxt == start:
                            # Cycle length is depth + 1
                            if depth + 1 < g:
                                return False
                        else:
                            idx = curr * n + nxt
                            if visit_id[idx] != current_id:
                                visit_id[idx] = current_id
                                new_queue.append((curr, nxt, depth + 1))
                queue = new_queue
            current_id += 1
            if current_id > 2**31:
                visit_id[:] = 0
                current_id = 1
        return True

    def get_girth(adj, n, max_girth=10):
        # Returns the girth of the graph
        # If girth > max_girth, return max_girth+1
        # We can modify has_girth_at_least_g to return the min cycle length found
        
        # For simplicity, we will just use has_girth_at_least_g in the search loop.
        # But for local search, it's better to know the actual short cycle.
        
        # Let's implement a function that finds the shortest cycle length up to max_girth
        # This is similar to the check above but returns the length.
        
        visit_id = np.zeros(n * n, dtype=np.int32)
        current_id = 1
        min_cycle = max_girth + 1
        
        for start in range(n):
            queue = []
            for nb in adj[start]:
                idx = start * n + nb
                if visit_id[idx] != current_id:
                    visit_id[idx] = current_id
                    queue.append((start, nb, 1))
            
            while queue:
                new_queue = []
                for prev, curr, depth in queue:
                    if depth >= max_girth - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start:
                            cycle_len = depth + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                        else:
                            idx = curr * n + nxt
                            if visit_id[idx] != current_id:
                                visit_id[idx] = current_id
                                new_queue.append((curr, nxt, depth + 1))
                queue = new_queue
            current_id += 1
            if current_id > 2**31:
                visit_id[:] = 0
                current_id = 1
            if min_cycle == 3: # Can't get better
                return 3
        return min_cycle

    def random_regular_graph(n, k, rng):
        # Returns adjacency list
        # Use configuration model
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        rng.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        seen = [set() for _ in range(n)]
        
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if v in seen[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
            seen[u].add(v)
            seen[v].add(u)
            
        return adj

    def swap_edges(adj, n, k, rng):
        # Pick two random edges (u1, v1) and (u2, v2)
        # Swap to (u1, u2) and (v1, v2) or (u1, v2) and (v1, u2)
        # Must maintain regularity and simplicity
        
        # Get list of edges
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        if len(edges) < 2:
            return False
            
        e1_idx = rng.randint(0, len(edges) - 1)
        e2_idx = rng.randint(0, len(edges) - 2)
        if e2_idx >= e1_idx:
            e2_idx += 1
            
        u1, v1 = edges[e1_idx]
        u2, v2 = edges[e2_idx]
        
        # Check if they share a vertex
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return False
            
        # Remove old edges
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        # Try swap 1: (u1, u2) and (v1, v2)
        if u2 not in adj[u1] and v2 not in adj[v1]:
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            return True
        
        # Try swap 2: (u1, v2) and (v1, u2)
        if v2 not in adj[u1] and u2 not in adj[v1]:
            adj[u1].append(v2)
            adj[v2].append(u1)
            adj[v1].append(u2)
            adj[u2].append(v1)
            return True
            
        # Revert
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        
        return False

    # Main search loop
    # Start from moore bound, increase n by 2 (to keep n*k even)
    n_start = moore
    if (n_start * k) % 2 != 0:
        n_start += 1
        
    # We will try to find a graph for increasing n
    # But we should prioritize finding *any* valid graph first, then minimize n?
    # The score is n. So we want the smallest n.
    # However, if we can't find one for small n quickly, we might waste time.
    # Let's try n from n_start up to a limit, with a time budget for each.
    
    max_n_to_try = 500 # Upper bound for search
    
    for n in range(n_start, max_n_to_try + 1, 2):
        if time.time() > deadline - 10:
            break
            
        # Try to find a valid graph for this n
        attempts = 0
        max_attempts = 50 # Adjust based on time
        
        while attempts < max_attempts and time.time() < deadline - 5:
            adj = random_regular_graph(n, k, rng)
            if adj is None:
                attempts += 1
                continue
                
            # Check girth
            if has_girth_at_least_g(adj, n, g):
                # Found a valid graph!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            
            # Local search to improve girth
            # We will try to increase girth
            current_girth = get_girth(adj, n, g)
            if current_girth >= g:
                 edges = []
                 for u in range(n):
                     for v in adj[u]:
                         if u < v:
                             edges.append((u, v))
                 return n, edges
                 
            # Perform local search
            # If current_girth < g, try to break short cycles
            # This is a heuristic. We just do random swaps and check.
            # To be efficient, we only check girth every few swaps.
            
            for _ in range(1000): # Max swaps per attempt
                if time.time() > deadline - 5:
                    break
                if swap_edges(adj, n, k, rng):
                    # Check girth every 10 swaps
                    # Or just check if girth improved?
                    # Checking girth is expensive.
                    # Let's check every 50 swaps.
                    if _ % 50 == 0:
                        if has_girth_at_least_g(adj, n, g):
                            edges = []
                            for u in range(n):
                                for v in adj[u]:
                                    if u < v:
                                        edges.append((u, v))
                            return n, edges
            attempts += 1
            
    # If no graph found, return the best effort or a fallback
    # Fallback: LUW construction from seed
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
        if time.time() > deadline - 2:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
