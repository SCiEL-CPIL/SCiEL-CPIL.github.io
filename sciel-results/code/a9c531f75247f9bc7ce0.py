import random
import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Fallback: The seed program's logic
    def seed_construct(k, g):
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

    # Helper: Generate random 3-regular graph
    def gen_random_3_reg(n):
        # Use configuration model
        half_edges = [i for i in range(n) for _ in range(3)]
        random.shuffle(half_edges)
        edges = set()
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i+1]
            if u == v:
                # Retry or fix
                # Simple retry
                while u == v:
                    j = random.randint(0, len(half_edges)-1)
                    if j != i and j != i+1:
                        v = half_edges[j]
                        half_edges[j] = u
                        u = half_edges[i+1]
                        break
            if (u, v) in edges or (v, u) in edges:
                 # This configuration model might produce multi-edges. 
                 # For simplicity, if multi-edge, we might fail or need repair.
                 # Let's just return None if we detect issues easily?
                 # Better: Use a library or more careful generation.
                 pass
            edges.add((min(u,v), max(u,v)))
        
        # Check degree
        deg = [0]*n
        for u,v in edges:
            deg[u]+=1
            deg[v]+=1
        if any(d!=3 for d in deg):
            return None
        if len(edges) != n*3//2:
            return None
        return list(edges)

    # Helper: Compute girth
    def compute_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = float('inf')
        for i in range(n):
            dist = [-1]*n
            parent = [-1]*n
            dist[i] = 0
            queue = [i]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and parent[v] != u:
                        # Found a cycle
                        # Reconstruct length
                        # This is a bit complex. 
                        # Simpler: BFS from each vertex to find shortest cycle back to start?
                        # No, standard girth algorithm:
                        # For each edge (u,v), if dist[u] + dist[v] + 1 < current min, update.
                        # But we need to be careful with the tree.
                        pass
        
        # Simpler girth check:
        # For each vertex, BFS. If we see a visited vertex that is not parent, 
        # we found a cycle. The length is dist[u] + dist[v] + 1.
        # We need the minimum such length.
        
        min_girth = float('inf')
        for i in range(n):
            dist = [-1]*n
            parent = [-1]*n
            dist[i] = 0
            queue = [i]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        # The cycle length is dist[u] + dist[v] + 1
                        # But this might not be the shortest cycle involving i?
                        # Actually, BFS ensures shortest path from i.
                        # So dist[u] + dist[v] + 1 is the length of the cycle formed by 
                        # path i->u, edge u-v, path v->i.
                        # Is it the shortest cycle in the graph? No.
                        # But the shortest cycle in the graph must be found by some BFS.
                        # We take the min over all edges.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    # This girth calculation is O(n^2) or O(n * (n+m)). For n=512, it's fine.
    # For n=10000, it might be slow.
    
    # Local search
    def improve_girth(n, edges, target_girth, time_limit):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        current_girth = compute_girth(n, edges)
        
        while current_girth < target_girth and time.time() < time_limit:
            # Find a short cycle
            # We can use the BFS info to find one.
            # For simplicity, let's just pick a random edge and see if it's in a short cycle?
            # Or find the shortest cycle explicitly.
            
            # Finding shortest cycle:
            # We can modify compute_girth to also return the cycle.
            # For now, let's just try random edge swaps.
            
            # Pick a random edge (u, v)
            u, v = random.choice(edges)
            # Remove it
            edges.remove((min(u,v), max(u,v)))
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Find new edge (u2, v2)
            # u2 is a neighbor of u? No, u has degree 2 now.
            # We need to connect u to some x, and v to some y, such that x-y is an edge?
            # No, 2-switch: remove (u,v) and (w,x), add (u,w) and (v,x).
            # This preserves degree.
            
            # Let's try a 2-switch.
            # Pick another edge (w, x) randomly.
            w, x = random.choice(edges)
            if w == u or w == v or x == u or x == v:
                # Revert
                edges.append((min(u,v), max(u,v)))
                adj[u].append(v)
                adj[v].append(u)
                continue
            
            # Remove (w, x)
            edges.remove((min(w,x), max(w,x)))
            adj[w].remove(x)
            adj[x].remove(w)
            
            # Add (u, w) and (v, x)
            # Check if these create loops or multi-edges
            if u == w or v == x or u == x or v == w:
                # Revert
                edges.append((min(w,x), max(w,x)))
                adj[w].append(x)
                adj[x].append(w)
                edges.append((min(u,v), max(u,v)))
                adj[u].append(v)
                adj[v].append(u)
                continue
            
            # Check if (u,w) or (v,x) already exist
            # We need a fast check. 
            # For small n, we can use a set of edges.
            edge_set = set((min(a,b), max(a,b)) for a,b in edges)
            if (min(u,w), max(u,w)) in edge_set or (min(v,x), max(v,x)) in edge_set:
                # Revert
                edges.append((min(w,x), max(w,x)))
                adj[w].append(x)
                adj[x].append(w)
                edges.append((min(u,v), max(u,v)))
                adj[u].append(v)
                adj[v].append(u)
                continue
            
            # Apply
            edges.append((min(u,w), max(u,w)))
            edges.append((min(v,x), max(v,x)))
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            # Check girth
            new_girth = compute_girth(n, edges)
            if new_girth <= current_girth:
                # Revert
                edges.remove((min(u,w), max(u,w)))
                edges.remove((min(v,x), max(v,x)))
                adj[u].remove(w)
                adj[w].remove(u)
                adj[v].remove(x)
                adj[x].remove(v)
                
                edges.append((min(u,v), max(u,v)))
                edges.append((min(w,x), max(w,x)))
                adj[u].append(v)
                adj[v].append(u)
                adj[w].append(x)
                adj[x].append(w)
            else:
                current_girth = new_girth
                
        return current_girth, edges

    best_n = None
    best_edges = None
    
    # Try increasing n
    for n in range(512, 1000, 2):
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            edges = gen_random_3_reg(n)
            if not edges:
                continue
                
            # Check if it's already valid
            girth = compute_girth(n, edges)
            if girth >= g:
                return n, edges
                
            # Improve
            girth, edges = improve_girth(n, edges, g, deadline - 5)
            
            if girth >= g:
                return n, edges
            else:
                # Keep track of best
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                    
    # If no valid graph found, return seed
    return seed_construct(k, g)

construct(3, 16)
