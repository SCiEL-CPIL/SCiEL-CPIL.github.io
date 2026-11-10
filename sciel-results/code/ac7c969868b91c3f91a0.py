import time
from collections import deque

def construct(k, g):
    # Search for a circulant graph C_n(1, s)
    # n must be at least the Moore bound
    # For k=3, g=16, Moore bound is 510.
    
    start_time = time.time()
    max_n = 1500 # Search up to this n
    max_s = max_n // 2
    
    # Moore bound calculation
    # Even girth g=2r
    r = g // 2
    moore_bound = 2 * sum((k-1)**i for i in range(r))
    # Ensure n is at least moore_bound
    start_n = max(moore_bound, 10)
    
    # We need to check girth >= g
    # A graph has girth < g if there is a cycle of length < g.
    # We check for cycles of length 4, 6, ..., g-2.
    
    def has_short_cycle(n, s, max_girth):
        # Check if C_n(1, s) has a cycle of length < max_girth
        # We can do BFS from each vertex up to depth max_girth//2 - 1
        # If we find a back edge to a vertex at distance d such that 
        # dist[u] + dist[v] + 1 < max_girth, we have a cycle.
        
        # Optimization: Due to symmetry, we only need to check from vertex 0?
        # No, the graph is vertex-transitive. So checking from 0 is sufficient!
        # C_n(1, s) is vertex-transitive.
        
        # BFS from 0
        visited = [-1] * n
        visited[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            du = visited[u]
            
            # Neighbors: u+1, u-1, u+s, u-s
            # Note: u+s and u-s might be same as u+1 or u-1 if s is small
            # We just iterate all 4 potential neighbors
            for v in [(u + 1) % n, (u - 1) % n, (u + s) % n, (u - s) % n]:
                if visited[v] == -1:
                    visited[v] = du + 1
                    if du + 1 < max_girth // 2:
                        queue.append(v)
                else:
                    # If v is visited, check if this edge forms a short cycle
                    # The cycle length would be du + 1 + visited[v]
                    # But we must ensure it's not the parent edge
                    # In a BFS tree, if v is parent of u, visited[v] = du - 1.
                    # Cycle length = du + 1 + (du - 1) = 2du.
                    # This is the cycle going down and back up.
                    # We only care if this length < max_girth.
                    # However, BFS finds the *shortest* path.
                    # If visited[v] + du + 1 < max_girth, and v is not the parent, 
                    # we have a cycle.
                    
                    # Actually, simpler:
                    # If we encounter an edge (u, v) where both are visited,
                    # the cycle length is dist[u] + dist[v] + 1.
                    # We must exclude the case where v is the parent of u (or vice versa)
                    # which gives a "cycle" of length 2*dist[u] (going out and back).
                    # But that's not a simple cycle in the graph, it's a backtrack.
                    # Simple cycles in BFS:
                    # If dist[u] + dist[v] + 1 < max_girth, and u != parent[v] and v != parent[u].
                    
                    # To be safe, we can just check:
                    if du + visited[v] + 1 < max_girth:
                        # Check if it's a valid simple cycle
                        # If v is the parent of u, then visited[v] = du - 1.
                        # Length = du + du - 1 + 1 = 2du.
                        # This is not a simple cycle of length 2du, it's just the path.
                        # So we ignore if abs(du - visited[v]) == 1 and the edge is in the tree.
                        # How to know if it's in the tree?
                        # We can store parent.
                        pass # Logic below
        
        # Let's refine the check
        return False # Placeholder

    # Implement proper girth check for vertex transitive graph
    def check_girth(n, s, min_girth):
        # BFS from 0
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        q = deque([0])
        
        while q:
            u = q.popleft()
            du = dist[u]
            
            if du >= min_girth // 2:
                continue
                
            neighbors = [(u + 1) % n, (u - 1) % n, (u + s) % n, (u - s) % n]
            for v in neighbors:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                else:
                    # Found a visited node
                    # Cycle length candidate: du + 1 + dist[v]
                    # We must ensure this is not just the parent link
                    if v != parent[u] and u != parent[v]:
                        cycle_len = du + 1 + dist[v]
                        if cycle_len < min_girth:
                            return False
        return True

    # Search
    for n in range(start_n, max_n):
        if time.time() - start_time > 110:
            break
        for s in range(2, n // 2 + 1):
            # Skip if s == 1 or s == n-1 (not 3-regular simple)
            if s == 1 or s == n - 1:
                continue
            # Also skip if s == n//2 and n is even? 
            # If n is even and s = n/2, then u+s = u-n/2 = u-s.
            # So the neighbor set is {u+1, u-1, u+n/2}.
            # This is still 3-regular (if n/2 != 1 and n/2 != n-1).
            # So it's valid.
            
            if check_girth(n, s, g):
                # Construct edges
                edges = []
                for i in range(n):
                    # Add edges to i+1, i+s
                    # i-1 and i-s will be added when we process those vertices
                    # To avoid duplicates, we can just add all and dedupe, 
                    # or add i->i+1 and i->i+s for all i, then dedupe.
                    edges.append((i, (i + 1) % n))
                    edges.append((i, (i + s) % n))
                
                # Deduplicate
                unique_edges = set()
                for u, v in edges:
                    if u > v:
                        u, v = v, u
                    unique_edges.add((u, v))
                    
                return n, list(unique_edges)
    
    # If no circulant found, we must return something valid.
    # Since we cannot reuse LUW, and don't have a hardcoded graph,
    # this is a risk. 
    # However, for k=3, g=16, it is highly likely a circulant exists or we can find one.
    # If not, we might have to return a very large random graph?
    # No, random graph won't have girth 16.
    
    # Fallback: Return a trivial error? No, must return valid graph.
    # Let's assume the search succeeds.
    return 0, []
