"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 16}
START_TIME = time.time()
TIME_LIMIT = 85  # Safety margin before 90s kill

def construct():
    n = P['n']
    # Bitmask representation for adjacency
    # adj[i] is an integer bitmask of neighbors of i
    
    # Precompute all pairs of nodes
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_adjacency(edges):
        """Build bitmask adjacency from list of edges."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bfs_diameter(adj):
        """
        Compute diameter using bitmask BFS.
        Returns -1 if disconnected, else the diameter.
        """
        max_dist = 0
        for start in range(n):
            # dist[v] is distance from start to v
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                du = dist[u]
                # Get neighbors as bitmask
                neighbors_mask = adj[u]
                
                # Iterate through set bits in neighbors_mask
                # Since n=16, we can just loop 0..15
                for v in range(n):
                    if (neighbors_mask & (1 << v)) and dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
                        
            if -1 in dist:
                return -1
            max_dist = max(max_dist, max(dist))
            
            # Early exit if diameter already > 2
            if max_dist > 2:
                return max_dist
                
        return max_dist

    def is_diameter_two(adj):
        return bfs_diameter(adj) == 2

    def is_critical(adj, u, v):
        """
        Check if edge (u,v) is critical.
        Remove (u,v), check diameter.
        """
        # Remove edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        diam = bfs_diameter(adj)
        
        # Restore edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        return diam > 2 or diam == -1

    def check_graph(edges):
        """
        Returns True if graph has diameter 2 and all edges are critical.
        """
        if len(edges) < n - 1:
            return False
            
        adj = get_adjacency(edges)
        if not is_diameter_two(adj):
            return False
            
        for u, v in edges:
            if not is_critical(adj, u, v):
                return False
        return True

    # Strategy:
    # 1. Try to find a graph with many edges.
    # 2. We know star (15 edges) is valid.
    # 3. Try to add edges to the star? Or search from high edge counts down?
    #    Searching from high edge counts down is risky because we might miss
    #    the optimal one if we don't check all combinations.
    #    However, for n=16, total edges is 120.
    #    
    #    Let's use a heuristic search but with bitmasks for speed.
    #    Start with a dense random graph? No, dense graphs usually have non-critical edges.
    #    Start with a sparse graph (tree) and add edges if they remain critical.
    
    # We will try to construct a graph by starting with a spanning tree
    # and greedily adding edges that are critical.
    # To maximize edges, we should try different spanning trees or different addition orders.
    
    best_edges = []
    best_score = 0
    
    # Since n=16 is small, we can try a few random restarts of greedy construction.
    # Greedy: Start with random spanning tree.
    # Try to add each missing edge. If adding it keeps all existing edges critical
    # AND the new edge is critical, add it.
    # Repeat for all missing edges.
    # This is a local search but deterministic within the greedy pass.
    
    import random
    
    def greedy_build():
        # Random spanning tree
        perm = list(range(n))
        random.shuffle(perm)
        edges = []
        for i in range(1, n):
            u = perm[i]
            v = random.choice(perm[:i])
            edges.append((u, v))
        
        current_adj = get_adjacency(edges)
        if not is_diameter_two(current_adj):
            # Spanning trees can have diameter > 2.
            # We need diameter 2.
            # A random tree might have diameter > 2.
            # If so, we can't start with it unless we fix it.
            # Actually, if tree diameter > 2, we need to add edges to reduce it.
            # But adding edges might make previous edges non-critical.
            # Let's just return invalid if initial tree doesn't have diameter 2?
            # No, we can add edges to reduce diameter.
            # But the condition is "Diameter exactly 2".
            # If tree diam > 2, we add edges until diam <= 2.
            # Then check criticality.
            
            # For simplicity, let's restart if tree diam > 2.
            # Most random trees on 16 nodes have diameter < 8, often 4-6.
            # Only stars and double-stars have diameter 2 or 3.
            # So random trees rarely have diameter 2.
            # This approach is bad.
            
            # Better approach: Start with a known diameter 2 graph (Star).
            # Then try to add edges.
            pass
            
        # Let's force start with Star to guarantee Diam 2.
        # Star: 0 connected to 1..15.
        edges = [(0, i) for i in range(1, n)]
        current_adj = get_adjacency(edges)
        
        # Now try to add edges between leaves (1..15)
        # If we add (u, v), does it break criticality of existing edges?
        # Existing edges are (0, x).
        # If we add (u, v), edge (0, u) might become non-critical if u still has diameter 2 paths to all others without (0, u).
        # In Star, (0, u) is critical because removing it disconnects u.
        # If we add (u, v), u is still connected to 0 via v? No, v is a leaf.
        # Path u-v-0? Yes.
        # So if we add (u, v), then (0, u) is no longer a bridge.
        # Is (0, u) still critical for diameter?
        # If we remove (0, u), u is connected to v (via edge u-v), v to 0 (via v-0).
        # So u is connected.
        # Dist(u, w) for any other leaf w?
        # u-v-0-w (dist 3).
        # So diameter becomes 3.
        # Thus (0, u) IS critical for diameter.
        
        # So we can add edges between leaves!
        # Let's try to add as many leaf-leaf edges as possible while keeping all edges critical.
        
        # Current edges: Star.
        # Candidate edges: (i, j) for 1 <= i < j <= 15.
        
        # We will iterate through candidate edges.
        # For each, check if adding it keeps the graph valid (Diam 2, all edges critical).
        # If yes, add it.
        
        # To maximize, we might want to be smart about order.
        # But let's just try all.
        
        # We need to check criticality of ALL edges after each addition.
        # This is expensive: O(E * E * BFS).
        # E starts at 15, grows. Max 120.
        # 120 * 120 * BFS(16) is manageable?
        # BFS(16) is very fast.
        
        valid = True
        # Check initial star
        if not check_graph(edges):
             # Star should always be valid.
             # 0 is center.
             # Remove (0, i): i disconnected. Critical.
             # Diam is 2.
             pass
            
        candidates = [(i, j) for i in range(1, n) for j in range(i+1, n)]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (u, v) in edges:
                continue
            # Try adding (u, v)
            new_edges = edges + [(u, v)]
            if check_graph(new_edges):
                edges = new_edges
                current_adj = get_adjacency(edges)
                
        return edges

    # Run multiple greedy restarts with different random candidate orders
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        candidate_edges = greedy_build()
        if len(candidate_edges) > best_score:
            best_score = len(candidate_edges)
            best_edges = candidate_edges
            
    # Fallback to star if nothing found (should not happen)
    if best_score == 0:
        return [[0, v] for v in range(1, n)]
        
    # Convert to list of lists of ints
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
        
    return result

# EVOLVE-BLOCK-END
