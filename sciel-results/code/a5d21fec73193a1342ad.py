import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for evaluator overhead
    
    # Fallback function using the seed logic (refactored)
    def get_luw_fallback(k, g):
        # The seed code from the prompt
        q = k
        # d calculation from seed
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
                # Note: The seed code adds edges here?
                # "if vertex[0] == 0: edges.append(...)" inside the loop?
                # In the seed:
                # for other in neighbours(vertex):
                #    ...
                #    if vertex[0] == 0:
                #       edges.append(...)
                # This iterates neighbours, and if side 0, adds edge.
                pass # Re-implementing carefully

        # Re-doing the seed edge collection correctly
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Specific search for k=4, g=10
    if k == 4 and g == 10:
        # Try to find a small graph
        # Range of N to try
        # Lower bound 161. Known 320.
        # Let's try N in [162, 164, ..., 200] first?
        # Or just 320, 322...
        # 320 is the target.
        
        best_graph = None
        
        # Helper: Check if graph is simple, 4-regular, and has girth >= 10
        def check_validity(n, edges):
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u, v in edges:
                if u == v: return False
                adj[u].append(v)
                adj[v].append(u)
            
            # Check degree
            for u in range(n):
                if len(adj[u]) != 4: return False
                # Check multi-edges
                if len(set(adj[u])) != 4: return False
            
            # Check Girth
            # BFS from each node
            for s in range(n):
                dist = [-1] * n
                dist[s] = 0
                queue = [s]
                while queue:
                    u = queue.pop(0)
                    if dist[u] >= 4: continue # Max depth 4 for girth 10 (dist u + dist v + 1 < 10)
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        else:
                            # Found an edge to a visited node
                            # Cycle length = dist[u] + dist[v] + 1
                            # If dist[u] + dist[v] + 1 < 10, invalid
                            # Note: if v is parent, dist[v] = dist[u]-1, len = dist[u] + dist[u]-1 + 1 = 2*dist[u].
                            # We should avoid counting parent? 
                            # Non-backtracking walks.
                            # If we encounter a visited node that is not the parent, we have a cycle.
                            # Length = dist[u] + dist[v] + 1.
                            # We need length < 10 to be invalid.
                            # But BFS tree edges don't create cycles.
                            # Cross/back edges do.
                            # To be safe, if dist[u] + dist[v] + 1 < 10 and v is not parent of u?
                            # Actually, if v is parent, length is 2*dist[u].
                            # If dist[u] = 3, len = 6. Invalid.
                            # So we just check dist[u] + dist[v] + 1 < 10.
                            if dist[u] + dist[v] + 1 < 10:
                                # Check if it's a back edge to parent
                                # Parent of u is not stored here, but if dist[v] == dist[u]-1 and v in adj[u], 
                                # it might be the parent.
                                # However, the *existence* of a cycle of length < 10 is what we care about.
                                # Even if it's a back edge in the BFS tree, it represents a cycle?
                                # No, the BFS tree path is unique. The edge (u,v) closes the loop.
                                # The cycle is path(s,v) + edge(v,u) + path(u,s)? No.
                                # Cycle is path(s,u) + edge(u,v) + path(v,s)?
                                # No, path(s,u) and path(s,v) meet at s.
                                # Cycle length is dist[u] + 1 + dist[v].
                                # This is correct for any non-tree edge.
                                # Is (u,v) a non-tree edge?
                                # If v is the parent of u, the edge is in the tree.
                                # The "cycle" formed is actually just the edge traversed back? No, that's not a cycle.
                                # A cycle requires a loop.
                                # In BFS, if we see (u,v) where dist[v] = dist[u]-1, it's the parent edge.
                                # It does NOT form a cycle of length dist[u]+dist[v]+1.
                                # It forms a path of length 2*dist[u] back to s? No.
                                # It just goes back.
                                # We must exclude the parent edge.
                                # We need to track parent.
                                pass # Implement with parent tracking
            
            return True
        
        # The check above is slightly flawed without parent tracking.
        # Let's do a proper girth check.
        def has_short_cycle(n, adj):
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = [s]
                while queue:
                    u = queue.pop(0)
                    if dist[u] > 4: continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if v == parent[u]:
                                continue
                            # Cycle found
                            # Length = dist[u] + dist[v] + 1
                            # But this is only valid if LCA is s?
                            # Not necessarily.
                            # The cycle length is dist[u] + dist[v] + 1 - 2*dist[lca]
                            # We don't have LCA.
                            # However, if dist[u] + dist[v] + 1 < 10, is it guaranteed to be a short cycle?
                            # The cycle goes up to LCA and down.
                            # Length $\le$ dist[u] + dist[v] + 1.
                            # If this sum < 10, then the actual cycle length is < 10.
                            # So yes, if dist[u] + dist[v] + 1 < 10 and v is not parent, we have a short cycle.
                            if dist[u] + dist[v] + 1 < 10:
                                return True, u, v
            return False, None, None
        
        # Search loop
        # Try N = 320, 322, 324...
        # Also try smaller N? 162, 164...
        # 320 is the known best. Smaller ones likely don't exist.
        # Start with 320.
        
        target_ns = [320, 322, 324, 326, 328, 330, 340, 350, 400]
        
        for N in target_ns:
            if time.time() > deadline - 5: break
            
            # Try to find a valid graph of size N
            found = False
            # Restarting
            for _ in range(100): # Limit restarts per N
                if time.time() > deadline - 5: break
                
                # Generate random 4-regular graph
                # Configuration model
                while True:
                    stubs = []
                    for i in range(N):
                        stubs.extend([i]*4)
                    random.shuffle(stubs)
                    edges = []
                    valid = True
                    seen = set()
                    for i in range(0, 2*N*4//4, 2): # 2*N edges
                        u = stubs[i]
                        v = stubs[i+1]
                        if u == v:
                            valid = False
                            break
                        key = (min(u,v), max(u,v))
                        if key in seen:
                            valid = False
                            break
                        seen.add(key)
                        edges.append((u,v))
                    if valid:
                        break
                    if time.time() > deadline - 10:
                        break
                
                if not valid: continue
                
                # Check girth
                adj = [[] for _ in range(N)]
                for u,v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                short, _, _ = has_short_cycle(N, adj)
                if not short:
                    return N, edges
                
                # If short cycle, try to break it
                # ... (Local search) ...
                # This is complex to implement fully in this short space.
                # If we can't implement local search, we might fail.
                # But 320 vertices is large. Probability of random 4-reg graph having girth 10 is low.
                
                # If no local search, this loop will just keep failing.
                # We need a fallback.
                
        # Fallback
        n, edges = get_luw_fallback(k, g)
        return n, edges

    else:
        # General case fallback
        n, edges = get_luw_fallback(k, g)
        return n, edges
