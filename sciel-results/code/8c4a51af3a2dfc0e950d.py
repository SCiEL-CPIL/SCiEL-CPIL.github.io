import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 80.0
    
    # Build base adjacency using bitmasks for speed
    adj_base = [0] * n
    edge_set = set()
    for u, v in base_edges:
        adj_base[u] |= (1 << v)
        adj_base[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
    
    # Generate all candidate edges (even-odd pairs not already in base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in edge_set:
                candidates.append((u, v))
    
    def get_diameter(adj_mask):
        """Compute diameter using bitmask BFS. Returns diameter or -1 if disconnected."""
        max_dist = 0
        full_mask = (1 << n) - 1
        for start in range(n):
            visited = 1 << start
            frontier = 1 << start
            dist = 0
            while frontier:
                dist += 1
                new_frontier = 0
                f = frontier
                while f:
                    lb = f & (-f)
                    f ^= lb
                    v = lb.bit_length() - 1
                    nbrs = adj_mask[v] & ~visited
                    new_frontier |= nbrs
                if not new_frontier:
                    break
                visited |= new_frontier
                frontier = new_frontier
            if visited != full_mask:
                return -1
            max_dist = max(max_dist, dist - 1)
        return max_dist

    def is_valid(added_edges_set):
        """Check if the set of added edges results in a valid graph."""
        adj_test = adj_base[:]
        for u, v in added_edges_set:
            adj_test[u] |= (1 << v)
            adj_test[v] |= (1 << u)
        
        # Check triangle-free
        # A triangle exists if for any edge (u,v), adj[u] & adj[v] != 0
        # We only need to check edges in the final graph?
        # Actually, if we add (u,v) and there is a path u-w-v of length 2, it's a triangle.
        # This is equivalent to adj[u] & adj[v] having a common neighbor.
        # Since we are adding edges to a bipartite graph (path is bipartite, we only add even-odd),
        # the graph remains bipartite?
        # Path is bipartite (Even, Odd).
        # We only add edges between Even and Odd.
        # So the graph remains bipartite.
        # Bipartite graphs are triangle-free by definition.
        # So we don't need to check for triangles explicitly if we ensure we only add even-odd edges.
        
        diam = get_diameter(adj_test)
        return diam <= 4

    # Start with the hub solution (10 edges)
    hub = 11
    initial_edges = set()
    for u in range(n):
        if u % 2 == 0 and (u, hub) not in edge_set and (hub, u) not in edge_set:
            initial_edges.add((min(u, hub), max(u, hub)))
    
    current_edges = initial_edges.copy()
    
    # Local Search
    # Try to remove edges
    while time.time() < deadline:
        improved = False
        # Try removing each edge
        for e in list(current_edges):
            if time.time() > deadline:
                break
            new_edges = current_edges - {e}
            if is_valid(new_edges):
                current_edges = new_edges
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps
        # Randomly swap an edge to see if we can maintain validity while changing structure
        # This helps escape local minima where removal fails but a swap might allow future removals
        # Or simply find a different configuration with same size but maybe easier to reduce?
        # Actually, if we can't remove, we are at a local minimum for that size.
        # We should try to find a solution with fewer edges by random restarts or different initializations.
        
        # Let's try a few random restarts with different hubs or random starts
        # But local search is fast. Let's try swapping to see if we can get stuck in a "worse" local minimum?
        # No, we want to minimize.
        
        # If we can't remove, let's try to find a better starting point.
        # Generate a few random valid solutions?
        # Or try different hubs.
        
        # Try hub at 12
        if not hasattr(construct, '_tried_hubs'):
            construct._tried_hubs = {11}
        
        if 12 not in construct._tried_hubs:
            construct._tried_hubs.add(12)
            hub = 12
            new_initial = set()
            for u in range(n):
                if u % 2 == 0 and (u, hub) not in edge_set and (hub, u) not in edge_set:
                    new_initial.add((min(u, hub), max(u, hub)))
            if is_valid(new_initial):
                current_edges = new_initial.copy()
                continue # Restart loop with new hub
        
        # If we are stuck, try random restarts with random edges
        # This is risky but might find a smaller set
        # Let's try one random restart
        if not hasattr(construct, '_restart_count'):
            construct._restart_count = 0
        
        if construct._restart_count < 10:
            construct._restart_count += 1
            # Random start: pick 8 random candidates, check validity, if valid, start local search
            # But random 8 might not be valid.
            # Better: start with a valid one and do random swaps to explore.
            # Since we are stuck, let's just break and return current best.
            # Or try to swap one edge to see if we can get to a different local min.
            # For now, break.
            break
        else:
            break
            
    # Convert to list of lists
    result = [[u, v] for u, v in current_edges]
    return result
