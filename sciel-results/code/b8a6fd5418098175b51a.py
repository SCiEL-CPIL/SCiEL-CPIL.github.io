import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    # Initialize adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute all candidate edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))
                
    def is_triangle_free(u, v):
        # Check if adding edge (u, v) creates a triangle
        # A triangle is formed if there exists a common neighbor w
        common = adj[u] & adj[v]
        return len(common) == 0

    def get_diameter():
        # BFS from all nodes to find diameter
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if any(d == -1 for d in dist):
                return 999
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam

    current_diam = get_diameter()
    added_edges = []
    
    # Greedy Set Cover approach:
    # We want to reduce diameter to <= 4.
    # While current_diam > 4, find the edge that reduces the diameter the most.
    # If multiple edges reduce it by the same amount, pick the one that "covers" the most far-apart pairs.
    
    # Optimization: Instead of full diameter calc for every candidate, we can be smarter,
    # but with n=24, full BFS is fast.
    # However, we must ensure we don't get stuck.
    
    # To handle potential local minima or specific structural requirements, 
    # we will also try a "hub" heuristic if greedy gets stuck or takes too long, 
    # but the primary driver is the greedy diameter reduction.
    
    # Let's refine the greedy step:
    # 1. Calculate current diameter D.
    # 2. If D <= 4, stop.
    # 3. For each valid candidate edge (triangle-free, not added, not base):
    #    - Tentatively add it.
    #    - Calculate new diameter D'.
    #    - Remove it.
    # 4. Select the edge with the smallest D'. If tie, select the one that minimizes the sum of squares of distances (more compact).
    # 5. Add the selected edge.
    # 6. Repeat.
    
    # To speed up, we can limit the search space if time is tight, but n=24 is small.
    # Number of candidates is ~276. BFS is O(N+E). 276 * 24 * 24 is trivial.
    
    while current_diam > 4:
        if time.time() > deadline:
            break
            
        best_edge = None
        best_new_diam = current_diam
        best_score = float('inf')
        
        # Sort candidates by potential? No, just iterate.
        for u, v in candidates:
            # Check if edge is already added
            if (u, v) in {(min(a, b), max(a, b)) for a, b in added_edges}:
                continue
                
            if not is_triangle_free(u, v):
                continue
                
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            
            # Calculate new diameter
            new_diam = get_diameter()
            
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (u, v)
                # Secondary metric: sum of squared distances to break ties
                # We can compute this if needed, but diameter reduction is primary.
                # Let's just pick the first one that achieves the min diameter to save time.
                # Actually, let's keep the one that minimizes diameter, and if tie, we can pick arbitrarily or use a heuristic.
                # For simplicity and speed, we break on first best found? No, we need the global best in this step.
                # But since we are doing a full scan, we keep the best found so far.
                
            elif new_diam == best_new_diam and best_edge is not None:
                # Tie-breaking: prefer edges that connect vertices with higher current distance?
                # Or just keep the first one.
                pass
                
        if best_edge is None:
            # No edge reduces diameter? This implies we might be in a local minimum 
            # or the target diameter is unreachable with triangle-free constraint from this state.
            # However, the problem guarantees a solution exists.
            # If we can't reduce diameter, maybe we need to accept a slightly higher diameter?
            # No, the contract says "diameter at most four".
            # If greedy fails to find a reduction, it might be because the *single* best edge doesn't reduce it,
            # but a *set* of edges does. This is a weakness of pure greedy.
            # Fallback: Use the known hub strategy which is guaranteed to work.
            # Let's switch to the hub strategy if greedy stalls.
            
            # Hub strategy:
            # Clear added edges and apply the proven hub logic.
            added_edges = []
            # Re-init adj to base
            for i in range(n):
                adj[i] = set()
            for u, v in base_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            hub = 11
            targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
            for v in targets:
                edge = (min(hub, v), max(hub, v))
                if edge not in base_set:
                    added_edges.append([hub, v])
                    adj[hub].add(v)
                    adj[v].add(hub)
            current_diam = get_diameter()
            break # Stop searching, return the hub solution
            
        # Commit the best edge
        u, v = best_edge
        added_edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
        current_diam = best_new_diam
        
    # Final verification
    if get_diameter() > 4:
        # Fallback to hub strategy if something went wrong
        added_edges = []
        for i in range(n):
            adj[i] = set()
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        hub = 11
        targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        for v in targets:
            edge = (min(hub, v), max(hub, v))
            if edge not in base_set:
                added_edges.append([hub, v])
                adj[hub].add(v)
                adj[v].add(hub)
                
    # Check triangle free one last time
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common = adj[u] & adj[v]
                if common:
                    # Triangle found, invalid. Fallback to hub.
                    added_edges = []
                    for i in range(n):
                        adj[i] = set()
                    for u2, v2 in base_edges:
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        
                    hub = 11
                    targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
                    for v2 in targets:
                        edge = (min(hub, v2), max(hub, v2))
                        if edge not in base_set:
                            added_edges.append([hub, v2])
                            adj[hub].add(v2)
                            adj[v2].add(hub)
                    break

    return [[int(u), int(v)] for u, v in added_edges]

# EVOLVE-BLOCK-END
