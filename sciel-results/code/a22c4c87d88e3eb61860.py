import time
import random
import numpy as np

def construct(k, g):
    # Target number of vertices
    # The best known is 1116. We try to construct a graph with this many vertices.
    # If it fails, we might try slightly larger, but the seed is ~1250.
    # We aim for N = 1116.
    
    start_time = time.time()
    time_limit = 110  # Leave some buffer
    
    # Try different N values starting from the best known
    # The Moore bound is 425, but it's likely not achievable.
    # We try 1116 first.
    target_N = 1116
    
    # We will use a greedy approach with randomization
    # Graph representation: adjacency list
    # For fast cycle checking, we might need to run BFS
    
    def is_valid_edge(u, v, adj, girth):
        # Check if adding edge (u, v) creates a cycle of length < girth
        # This is equivalent to checking if the distance between u and v in the current graph is < girth - 1
        # If dist(u, v) < girth - 1, then adding (u, v) creates a cycle of length dist(u, v) + 1 < girth
        
        # BFS from u
        if u == v:
            return False
        
        # If they are already connected, it's a multi-edge (not allowed) or self-loop
        if v in adj[u]:
            return False
            
        # BFS
        from collections import deque
        visited = {u: 0}
        queue = deque([u])
        
        while queue:
            curr = queue.popleft()
            dist = visited[curr]
            if dist >= girth - 1:
                break
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    if neighbor == v:
                        # Found v at distance dist + 1
                        # Cycle length will be dist + 1 + 1 = dist + 2
                        # We need dist + 2 >= girth
                        # So we need dist >= girth - 2
                        if dist + 2 < girth:
                            return False
                        else:
                            return True
                    visited[neighbor] = dist + 1
                    queue.append(neighbor)
        
        # If v was not reached within girth - 2 steps, it's safe
        return True

    def try_construct(N, k, g, seed):
        random.seed(seed)
        adj = [[] for _ in range(N)]
        degrees = [0] * N
        total_edges_needed = N * k // 2
        edges_added = 0
        
        # Initialize with a random matching to start? Or just greedy from empty.
        # Greedy from empty is slow because early vertices have few options.
        # Better: Start with a random regular graph and repair?
        # Or: Greedy with high priority on low-degree vertices.
        
        # We'll use a greedy approach.
        # To avoid getting stuck, we use a "cooling" factor or restarts.
        
        for step in range(total_edges_needed):
            # Pick a vertex with the lowest degree
            # To speed up, we can maintain a list of vertices sorted by degree, 
            # but for N=1116, sorting every step is O(N log N) * N = O(N^2 log N) ~ 10^7 ops. Fine.
            
            # Find vertex with min degree
            min_deg = min(degrees)
            candidates_u = [i for i in range(N) if degrees[i] == min_deg]
            if not candidates_u:
                return None, None # Should not happen
            
            u = random.choice(candidates_u)
            
            # Find candidates for v
            # v must have degree < k
            # v must not be u
            # v must not be already connected to u
            
            valid_vs = []
            for v in range(N):
                if v == u:
                    continue
                if degrees[v] >= k:
                    continue
                if v in adj[u]:
                    continue
                valid_vs.append(v)
            
            if not valid_vs:
                return None, None # Stuck
            
            # Shuffle valid_vs to add randomness
            random.shuffle(valid_vs)
            
            edge_added = False
            # Try up to 20 candidates
            for v in valid_vs[:20]:
                if is_valid_edge(u, v, adj, g):
                    # Add edge
                    adj[u].append(v)
                    adj[v].append(u)
                    degrees[u] += 1
                    degrees[v] += 1
                    edges_added += 1
                    edge_added = True
                    break
            
            if not edge_added:
                # Failed to find a valid edge for u
                return None, None

        # Verify final graph
        # Check degrees
        if any(d != k for d in degrees):
            return None, None
            
        # Check girth
        # We need to ensure no cycle < g
        # We can run BFS from every node? O(N * E) = 1116 * 2750 ~ 3M. Fine.
        # Or just trust the greedy check? The greedy check ensures that *at the time of addition*, no short cycle was formed.
        # Since we only add edges, and we check against the *current* graph, if we never form a short cycle, the final graph has no short cycles.
        # Wait, does adding an edge ever *remove* a cycle? No.
        # So if we ensure that every added edge does not create a short cycle in the current graph, the final graph has girth >= g.
        
        # Construct edge list
        edge_list = []
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))
                    
        return N, edge_list

    # Try to construct with N = 1116
    best_result = None
    # Try a few seeds
    for seed in range(10):
        if time.time() - start_time > time_limit:
            break
        n, edges = try_construct(target_N, k, g, seed)
        if n is not None:
            best_result = (n, edges)
            break # Found a valid graph with 1116 vertices
            
    if best_result is None:
        # If 1116 fails, try a slightly larger N to ensure we return *something* valid
        # The seed returns ~1250. We can try 1200.
        for N_try in [1200, 1250]:
            if time.time() - start_time > time_limit:
                break
            for seed in range(5):
                n, edges = try_construct(N_try, k, g, seed)
                if n is not None:
                    best_result = (n, edges)
                    break
            if best_result is not None:
                break
                
    if best_result is None:
        # Fallback to the seed logic if all else fails?
        # The seed is always valid. We can just call the seed logic.
        # But we are rewriting from scratch.
        # Let's just return a trivial graph if we fail? No, must be valid.
        # If we fail, we should probably use the seed's construction to guarantee a valid output.
        
        # Re-implement seed logic as fallback
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

    return best_result
