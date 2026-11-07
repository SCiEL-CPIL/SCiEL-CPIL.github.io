import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    # Build adjacency from base
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # All possible non-base edges
    all_possible = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_edges:
                all_possible.append((u, v))
    
    # Precompute: for each candidate edge (u,v), which other edges would create a triangle
    # Triangle (u,v,w) exists if w is adjacent to both u and v in the current graph
    # We need to check dynamically as we add edges
    
    def is_triangle_free(edge_set):
        """Check if the graph with base + edge_set is triangle-free."""
        # Build full adjacency
        full_adj = [set() for _ in range(n)]
        for u, v in base_edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        for u, v in edge_set:
            full_adj[u].add(v)
            full_adj[v].add(u)
        # Check for triangles
        for u in range(n):
            for v in full_adj[u]:
                if v <= u:
                    continue
                for w in full_adj[u]:
                    if w <= v:
                        continue
                    if w in full_adj[v]:
                        return False
        return True
    
    def compute_diameter(edge_set):
        """Compute diameter of graph with base + edge_set. Return -1 if not connected."""
        full_adj = [set() for _ in range(n)]
        for u, v in base_edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        for u, v in edge_set:
            full_adj[u].add(v)
            full_adj[v].add(u)
        
        # BFS from each vertex
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in full_adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if -1 in dist:
                return -1  # Not connected
            max_dist = max(max_dist, max(dist))
        return max_dist
    
    def is_valid(edge_set):
        """Check if edge_set makes graph triangle-free, connected, diameter <= 4."""
        if not is_triangle_free(edge_set):
            return False
        d = compute_diameter(edge_set)
        return d >= 0 and d <= 4
    
    # Try to find minimum number of edges
    # Lower bound: we need enough edges to reduce diameter from 23 to 4
    # Start with small numbers and increase
    
    deadline = time.time() + 80  # Safety margin
    
    best_solution = None
    
    # Try k = 4, 5, 6, ... edges
    for k in range(4, 15):
        if time.time() > deadline:
            break
        if best_solution is not None:
            break
        
        # Use greedy + local search or randomized approach
        # For small k, we can try smarter search
        
        # Strategy: use a heuristic to pick promising edges first
        # Edges that span large distances are more useful for reducing diameter
        
        # Sort candidates by distance (longer first)
        sorted_candidates = sorted(all_possible, key=lambda e: abs(e[0]-e[1]), reverse=True)
        
        # Use a beam search / greedy approach with backtracking for small k
        # For k <= 6, try a more systematic approach
        
        if k <= 6:
            # Try all combinations is too large, use greedy with restarts
            found = False
            for attempt in range(100):
                if time.time() > deadline:
                    break
                
                # Greedy: pick edges one by one, preferring those that reduce diameter most
                current_edges = []
                current_adj = [set() for _ in range(n)]
                for u, v in base_edges:
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                
                for _ in range(k):
                    # Find best edge to add
                    best_edge = None
                    best_score = -1
                    
                    # Sample candidates (or all if few)
                    candidates = sorted_candidates
                    if len(candidates) > 50:
                        import random
                        candidates = random.sample(candidates, 50)
                    
                    for u, v in candidates:
                        if (u, v) in base_edges:
                            continue
                        # Check if adding this edge creates a triangle
                        # Triangle if there exists w adjacent to both u and v
                        triangle = False
                        for w in current_adj[u]:
                            if w in current_adj[v]:
                                triangle = True
                                break
                        if triangle:
                            continue
                        
                        # Check diameter after adding
                        test_edges = current_edges + [(u, v)]
                        d = compute_diameter(test_edges)
                        if d > 4:
                            continue
                        
                        # Score: prefer edges that reduce diameter more
                        score = -d  # Lower diameter is better
                        if score > best_score:
                            best_score = score
                            best_edge = (u, v)
                    
                    if best_edge is None:
                        # No valid edge found, this attempt fails
                        break
                    
                    current_edges.append(best_edge)
                    u, v = best_edge
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                
                if len(current_edges) == k and is_valid(current_edges):
                    best_solution = current_edges
                    found = True
                    break
            
            if found:
                break
        else:
            # For larger k, use a different heuristic
            # Greedy with multiple restarts
            found = False
            for attempt in range(200):
                if time.time() > deadline:
                    break
                
                current_edges = []
                current_adj = [set() for _ in range(n)]
                for u, v in base_edges:
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                
                for _ in range(k):
                    best_edge = None
                    best_score = -1
                    
                    import random
                    candidates = random.sample(sorted_candidates, min(80, len(sorted_candidates)))
                    
                    for u, v in candidates:
                        if (u, v) in base_edges:
                            continue
                        triangle = False
                        for w in current_adj[u]:
                            if w in current_adj[v]:
                                triangle = True
                                break
                        if triangle:
                            continue
                        
                        test_edges = current_edges + [(u, v)]
                        d = compute_diameter(test_edges)
                        if d > 4:
                            continue
                        
                        score = -d
                        if score > best_score:
                            best_score = score
                            best_edge = (u, v)
                    
                    if best_edge is None:
                        break
                    
                    current_edges.append(best_edge)
                    u, v = best_edge
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                
                if len(current_edges) == k and is_valid(current_edges):
                    best_solution = current_edges
                    found = True
                    break
            
            if found:
                break
    
    if best_solution is not None:
        # Convert to list of [u, v]
        result = [[u, v] for u, v in best_solution]
        # Ensure types are correct
        return result
    else:
        # Fallback: return a known valid solution if we can find one quickly
        # This shouldn't happen, but provide a safe fallback
        # Try a simple construction: add edges (0,12), (0,23), (12,23), etc.
        fallback = []
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try adding strategic edges
        candidates = [(0, 12), (0, 23), (12, 23), (1, 13), (1, 22), (13, 23), (2, 14), (2, 21), (14, 23), (3, 15), (3, 20), (15, 23), (4, 16), (4, 19), (16, 23), (5, 17), (5, 18), (17, 23), (6, 18), (6, 17), (7, 19), (8, 20), (9, 21), (10, 22), (11, 23)]
        
        for u, v in candidates:
            if (u, v) in base_edges:
                continue
            # Check triangle
            triangle = False
            for w in adj[u]:
                if w in adj[v]:
                    triangle = True
                    break
            if triangle:
                continue
            fallback.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
        
        if is_valid(fallback):
            return [[u, v] for u, v in fallback]
        
        # Last resort: return empty (will be invalid, but better than exception)
        return []

# EVOLVE-BLOCK-END
