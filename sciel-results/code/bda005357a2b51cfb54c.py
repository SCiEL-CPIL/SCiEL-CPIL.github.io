import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Moore bound for k=3, g=16 is 510.
    # We try n from 510 upwards.
    # But we need to be careful: if we can't find one, we should return a valid one.
    # The LUW graph is a fallback.
    
    def luw_fallback(k, g):
        # Re-implement the seed's LUW logic here as a fallback
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

    def try_construct(n, k, g, seed):
        random.seed(seed)
        degrees = [0] * n
        edges = []
        adj = [[] for _ in range(n)]
        
        # Precompute all possible edges? No, too many.
        # Just pick random pairs.
        
        attempts = 0
        max_attempts = 100000 # per n
        
        while True:
            if time.time() > deadline:
                return None
            
            # Find vertices with degree < k
            candidates = [i for i in range(n) if degrees[i] < k]
            if not candidates:
                break
            
            u = random.choice(candidates)
            found = False
            # Shuffle candidates to avoid bias
            random.shuffle(candidates)
            for v in candidates:
                if u == v:
                    continue
                if degrees[v] >= k:
                    continue
                if v in adj[u]:
                    continue
                
                # Check if adding (u, v) creates a short cycle
                # This is equivalent to checking if dist(u, v) < g - 1
                # But we only have the current graph.
                # We can do a BFS from u up to depth g-2 and see if v is reached.
                # Or BFS from v.
                # Since g is 16, depth 14.
                # This is O(n) per check.
                
                # Optimization: if the graph is sparse, BFS is fast.
                visited = {u: 0}
                queue = [u]
                dist = 0
                found_short = False
                while queue and dist < g - 2:
                    next_queue = []
                    for node in queue:
                        for nei in adj[node]:
                            if nei == v:
                                found_short = True
                                break
                            if nei not in visited:
                                visited[nei] = dist + 1
                                next_queue.append(nei)
                    if found_short:
                        break
                    queue = next_queue
                    dist += 1
                
                if not found_short:
                    # Add edge
                    edges.append((u, v))
                    adj[u].append(v)
                    adj[v].append(u)
                    degrees[u] += 1
                    degrees[v] += 1
                    found = True
                    break
            
            if not found:
                return None # Failed to complete
            
            attempts += 1
            if attempts > max_attempts:
                return None
        
        # Verify final graph
        # Check degrees
        for i in range(n):
            if degrees[i] != k:
                return None
        # Check girth
        # We already ensured no cycle < g during construction, but let's verify.
        # Actually, the construction ensures that no NEW short cycle is created.
        # But could there be a short cycle formed by multiple edges?
        # No, because we check dist(u, v) < g-1.
        # If dist(u, v) >= g-1, then adding (u, v) creates a cycle of length >= g.
        # So the girth is at least g.
        
        return n, edges

    best_result = None
    best_n = float('inf')
    
    # Try n from Moore bound upwards
    # Moore bound for k=3, g=16 is 510.
    # We try a few values.
    for n in range(510, 1000, 2):
        if time.time() > deadline:
            break
        for seed in range(100):
            if time.time() > deadline:
                break
            result = try_construct(n, k, g, seed)
            if result:
                n_found, edges_found = result
                if n_found < best_n:
                    best_n = n_found
                    best_result = result
                # If we found a valid graph, we can return it immediately?
                # No, we might find a smaller one later?
                # But we are iterating n upwards. So the first valid one is the smallest.
                # So we can return immediately.
                return result
    
    # If no graph found, fall back to LUW
    return luw_fallback(k, g)
