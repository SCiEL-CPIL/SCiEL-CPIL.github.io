import random
import time

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # 5 seconds margin before 90s limit
    
    def is_valid(adj, deg):
        # Check if any simple path s-u-v-t has deg[s] == deg[t]
        # s != t, u != v, s != v, t != u, s != u (implied by simple path in graph context usually, 
        # but here "simple three-edge path" means vertices s,u,v,t are distinct? 
        # "simple three-edge path s-u-v-t" implies s,u,v,t are distinct vertices.
        # Let's verify the definition: "simple path" usually means no repeated vertices.
        # So s, u, v, t must be distinct.
        for u in range(n):
            for v in adj[u]:
                if v == u: continue
                for s in adj[u]:
                    if s == v: continue
                    if s == u: continue
                    for t in adj[v]:
                        if t == u: continue
                        if t == v: continue
                        if t == s: continue
                        # s, u, v, t are distinct
                        if deg[s] == deg[t]:
                            return False
        return True

    def try_add_edge(adj, deg, u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        
        # Tentatively add
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
        
        if is_valid(adj, deg):
            return True
        else:
            # Rollback
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
            return False

    # Initialize with an empty graph
    adj = [[] for _ in range(n)]
    deg = [0] * n
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # First pass: Greedy addition in random order
    random.shuffle(all_edges)
    for u, v in all_edges:
        if time.time() > deadline:
            break
        try_add_edge(adj, deg, u, v)
        
    current_edges = sum(len(a) for a in adj) // 2
    
    # Local search: Try to add more edges by shuffling and retrying
    # Or try to swap edges? The task is to maximize edge count.
    # Since we can only add edges (removing doesn't help maximize count directly unless it allows adding 2 more),
    # a simple greedy might get stuck. 
    # However, removing an edge increases degrees of 2 vertices by -1, potentially breaking equal-degree constraints?
    # No, changing degrees might *create* equalities or *break* them.
    # The constraint is: NO path s-u-v-t has deg[s]==deg[t].
    # If we remove an edge, degrees change. This might allow adding other edges.
    
    # Let's stick to a robust randomized greedy with multiple restarts or a more complex search.
    # Given the small N=15, we can try a more exhaustive local search.
    
    # Strategy: 
    # 1. Start empty.
    # 2. Repeatedly pick a random missing edge. If valid, add it.
    # 3. If stuck (no missing edge can be added), try to remove a random existing edge, then try to add different ones.
    
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    best_count = len(best_edges)
    
    while time.time() < deadline:
        # Try to add any missing edge
        added = False
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
        random.shuffle(missing)
        for u, v in missing:
            if try_add_edge(adj, deg, u, v):
                added = True
                break
        
        if not added:
            # Stuck. Try to remove a random edge to unlock new possibilities
            if best_count > 0:
                existing = [(i, j) for i in range(n) for j in adj[i] if i < j]
                if existing:
                    ru, rv = random.choice(existing)
                    # Remove
                    adj[ru].remove(rv)
                    adj[rv].remove(ru)
                    deg[ru] -= 1
                    deg[rv] -= 1
                    
                    # Try to add some edges
                    for _ in range(10): # Try a few random additions
                        if not missing:
                            missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
                            if not missing:
                                break
                        random.shuffle(missing)
                        for u, v in missing:
                            if try_add_edge(adj, deg, u, v):
                                # Update missing list if we added
                                missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
                                break
                        
                        if time.time() > deadline:
                            break
                        
                        if not missing:
                            break
                        
        # Update best
        current_count = sum(len(a) for a in adj) // 2
        if current_count > best_count:
            best_count = current_count
            best_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        best_edges.append([u, v])
            
        # If we've found a very high density, we might want to stop early?
        # Max edges is 105. 
        # Let's just run until deadline.
        
        # Optimization: If we can't add any edge and removing one doesn't help immediately, 
        # we might be in a local optimum. 
        # To avoid infinite loops of remove-add-same, we can track state or just rely on randomness.
        
        # A more aggressive strategy: 
        # If stuck, restart with a random subset of current edges?
        # Or just continue the random removal/addition process.
        
        # Let's ensure we don't waste time on very small graphs if we already have a good one.
        # But maximizing is the goal.
        
        # To prevent the "remove one, add one back" loop:
        # We can check if the graph state is the same as before removal.
        # Simple heuristic: if we removed an edge and didn't add a *different* set of edges to exceed previous count, 
        # the removal was bad. But we already track best_count.
        
        # Let's add a check: if current_count < best_count, we are in a worse state. 
        # We should try to recover or just let the loop continue.
        # Actually, if we remove an edge, we are in a lower state. 
        # The loop `if not added` triggers removal. 
        # If removal allows adding 2 edges, count increases.
        # If removal allows adding 1 edge, count stays same (removed 1, added 1).
        # If removal allows adding 0 edges, count decreases.
        
        # We should avoid decreasing count if possible.
        # But sometimes you must go down to go up.
        
        # To be safe, let's just run the loop. The `best_edges` tracks the maximum found.

    # Convert to list of lists of ints
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
        
    return result
