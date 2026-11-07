import time
import collections

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # Base adjacency list
    base_adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Precompute bipartite sets
    even_verts = set(i for i in range(n) if i % 2 == 0)
    odd_verts = set(i for i in range(n) if i % 2 == 1)
    
    def is_bipartite_edge(u, v):
        return (u in even_verts and v in odd_verts) or (u in odd_verts and v in even_verts)
        
    def get_diameter(added_edges):
        """
        Calculates diameter of the graph formed by base_edges + added_edges.
        Returns -1 if disconnected.
        """
        # Build adjacency
        adj = [list(nei) for nei in base_adj]
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_dist = 0
        # BFS from each node
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = collections.deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        q.append(v)
            if -1 in dist:
                return -1 # Disconnected
        return max_dist

    def is_valid_solution(added_edges):
        # Check triangle-free (bipartite)
        for u, v in added_edges:
            if not is_bipartite_edge(u, v):
                return False
            if (u, v) in base_edges:
                return False
        # Check diameter
        d = get_diameter(added_edges)
        return d != -1 and d <= 4

    # 1. Generate an initial valid solution using Greedy Addition
    added = []
    added_set = set()
    
    # Candidates: all bipartite edges not in base
    candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (u, v) not in base_edges and (v, u) not in base_edges:
                candidates.append((u, v))
                
    # Greedy: Add edge that reduces diameter the most
    while True:
        if time.time() > deadline:
            break
        current_diam = get_diameter(added)
        if current_diam <= 4:
            break
        if current_diam == -1:
            # Should not happen as base is connected
            break
            
        best_edge = None
        best_new_diam = current_diam + 1
        
        # To speed up, we can sample candidates or sort them by potential
        # But n=24 is small, candidates ~ 12*12 - 11 = 133.
        # Evaluating 133 edges * BFS (1152 ops) = 150k ops. Fast.
        
        for u, v in candidates:
            if (u, v) in added_set or (v, u) in added_set:
                continue
            # Quick check: does this edge help?
            # We can simulate adding it
            trial_added = added + [(u, v)]
            # Optimization: We don't need full diameter if we just want to see if it's better
            # But we need the value to compare.
            new_diam = get_diameter(trial_added)
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (u, v)
                
        if best_edge is None:
            # No improvement found? Should not happen if diam > 4
            break
            
        u, v = best_edge
        added.append((u, v))
        added_set.add((u, v))
        added_set.add((v, u))
        
    # If we didn't reach diameter 4, we have a problem. 
    # But for P24, it's always possible.
    
    # 2. Minimize: Prune edges
    # We have a valid set `added`. Try to remove edges.
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(added)):
            # Try removing added[i]
            trial_added = added[:i] + added[i+1:]
            if is_valid_solution(trial_added):
                added = trial_added
                improved = True
                break # Restart loop as indices changed
                
    # 3. Final Verification
    if not is_valid_solution(added):
        # Fallback to a known good solution if something went wrong
        # E.g. Hub 12 to all odds
        fallback = []
        for v in odd_verts:
            if (12, v) not in base_edges and (v, 12) not in base_edges:
                fallback.append((12, v))
        if is_valid_solution(fallback):
            added = fallback
        else:
            # Extreme fallback: return empty (invalid) or best effort
            pass

    # Return as list of lists
    return [[int(u), int(v)] for u, v in added]
