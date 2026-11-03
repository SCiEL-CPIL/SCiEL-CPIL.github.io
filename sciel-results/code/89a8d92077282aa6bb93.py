import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Moore bound
    # n >= 1 + k * sum((k-1)^i, i=0 to floor((g-1)/2))
    # For g=9, k=5: 1 + 5*(1+4+16+64) = 426
    
    # We will try to find a random graph with n < seed_n (6250)
    # If we fail, we fall back to the L-U-W construction from the seed.
    
    def get_seed_result():
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

    def generate_random_regular(n, k):
        """Generates a random k-regular graph on n vertices using configuration model."""
        if n * k % 2 != 0:
            return None
        # Create stubs
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                # Loop, try to fix by swapping
                # Simple repair: find another stub j to swap with
                # For simplicity, if loop, just return None and retry
                # But this is inefficient.
                # Let's just return None and hope the next random trial works.
                # Or implement a proper repair.
                # Given time, let's just retry generation.
                return None
            edges.append((u, v))
        
        # Check for multi-edges
        edge_set = set()
        for u, v in edges:
            if u > v: u, v = v, u
            if (u, v) in edge_set:
                return None
            edge_set.add((u, v))
        
        return edges

    def compute_girth(n, edges, max_girth):
        """Computes girth up to max_girth. Returns girth or max_girth if none found."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = max_girth
        # BFS from each vertex
        # Optimization: only need to check up to depth max_girth//2
        depth_limit = max_girth // 2
        
        for start in range(n):
            # visited: dict {node: (depth, parent)}
            visited = {start: (0, -1)}
            queue = [(start, 0, -1)]
            
            while queue:
                node, depth, parent = queue.pop(0)
                if depth >= depth_limit:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        # Cycle found
                        # depth of current + depth of neighbor + 1
                        # Note: visited[neighbor] is the first time we saw it.
                        # This BFS finds the shortest path from start to neighbor.
                        # If we encounter a back edge, the cycle length is depth + visited[neighbor][0] + 1
                        # This is only valid if the back edge is to an ancestor in the BFS tree?
                        # Actually, standard BFS finds shortest path.
                        # If we see a visited node, the cycle length is depth + visited[neighbor][0] + 1.
                        # This might not be the *shortest* cycle in the graph, but it finds *a* cycle.
                        # To find the *girth*, we need the minimum over all such cycles.
                        cycle_len = depth + visited[neighbor][0] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth == 3: # Triangle is minimal
                            return 3
                    else:
                        visited[neighbor] = (depth + 1, node)
                        queue.append((neighbor, depth + 1, node))
                        
        return min_girth

    # Try random search for n < 6250
    # We want n to be as small as possible.
    # Let's try a few n values.
    # If k=5, g=9, Moore bound is 426.
    # Let's try n in [500, 1000, 2000, 4000, 6000]
    # If we find a valid one, we return it.
    
    candidate_ns = [500, 600, 800, 1000, 1500, 2000, 3000, 5000, 6000]
    
    best_n = float('inf')
    best_edges = None
    
    for n in candidate_ns:
        if time.time() > deadline:
            break
            
        # Try a few random instances for this n
        attempts = 5
        for _ in range(attempts):
            if time.time() > deadline:
                break
            edges = generate_random_regular(n, k)
            if edges is None:
                continue
            girth = compute_girth(n, edges, g)
            if girth >= g:
                best_n = n
                best_edges = edges
                break # Found a valid graph for this n, try smaller n?
                # Actually, we iterate n in increasing order.
                # So if we find one at n=500, we should return it immediately?
                # Yes, we want minimum n.
                # But we should check if we can find a smaller one?
                # We are iterating from small to large.
                # So the first valid one is the smallest.
                return n, edges
        
    # If no random graph found, fall back to seed
    if time.time() > deadline:
        # If we are out of time, we must return *something* valid.
        # The seed is valid.
        # But computing the seed might take time?
        # The seed is deterministic and fast for k=5.
        pass
        
    # Compute seed result
    seed_n, seed_edges = get_seed_result()
    
    # If we found a random graph that is better than seed, use it
    if best_n < seed_n:
        return best_n, best_edges
    else:
        return seed_n, seed_edges
