import random
import time

def construct():
    n = 15
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    def get_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid(edges):
        if not edges:
            return True
        adj = get_adj(edges)
        deg = get_degrees(edges)
        for u, v in edges:
            # Check if there exists s in N(u)\{v} and t in N(v)\{u} with deg[s] == deg[t]
            # Optimization: Precompute set of degrees of neighbors
            degs_u = set(deg[x] for x in adj[u] if x != v)
            degs_v = set(deg[y] for y in adj[v] if y != u)
            if degs_u & degs_v:
                return False
        return True

    def count_conflicts(edges):
        """
        Returns a dict mapping edge -> number of conflicts it is involved in.
        An edge (u,v) is in conflict if there exists s,t such that deg(s)==deg(t).
        We count the number of such (s,t) pairs for each edge.
        """
        if not edges:
            return {}
        adj = get_adj(edges)
        deg = get_degrees(edges)
        conflicts = {e: 0 for e in edges}
        
        for u, v in edges:
            degs_u = [deg[x] for x in adj[u] if x != v]
            degs_v = [deg[y] for y in adj[v] if y != u]
            # Count pairs (s, t) where s in N(u), t in N(v) and deg[s] == deg[t]
            # We can just check if the sets intersect. But for "weight", count is better.
            # Actually, if they intersect, the edge is invalid.
            # To decide which to remove, maybe remove the one with most "potential" neighbors?
            # Let's just stick to random or most central.
            pass
        return conflicts

    def repair(edges):
        """
        Takes a list of edges (possibly invalid) and removes edges until valid.
        """
        edges = list(edges)
        while True:
            if is_valid(edges):
                break
            
            # Find all invalid edges
            adj = get_adj(edges)
            deg = get_degrees(edges)
            invalid_edges = []
            for u, v in edges:
                degs_u = set(deg[x] for x in adj[u] if x != v)
                degs_v = set(deg[y] for y in adj[v] if y != u)
                if degs_u & degs_v:
                    invalid_edges.append((u, v))
            
            if not invalid_edges:
                break # Should not happen if is_valid returned False

            # Strategy: Remove the edge that is "most central" or random.
            # Let's try removing the edge with the highest sum of degrees of its endpoints.
            # Or just random. Random with restarts is often robust.
            # Let's pick the one with max (deg[u] + deg[v]) to break strong connections?
            # Or min?
            # Let's try random first, it's simpler and often works with enough restarts.
            
            # Better: Pick the edge that participates in the most "bad" configurations?
            # Since we just know it's invalid, let's pick the one with max degree sum.
            # This tends to reduce density of high-degree nodes, which might help.
            
            # Actually, let's just pick random to avoid bias.
            edge_to_remove = random.choice(invalid_edges)
            edges.remove(edge_to_remove)
            
        return edges

    def local_search_add(edges):
        """
        Greedily add any edge that keeps the graph valid.
        """
        edges = set(edges)
        improved = True
        while improved:
            improved = False
            # Try to add each missing edge
            # Shuffle to avoid bias
            candidates = [e for e in all_edges if e not in edges]
            random.shuffle(candidates)
            for e in candidates:
                if is_valid(list(edges) + [e]):
                    edges.add(e)
                    improved = True
                    # Don't break, try to add more in same pass? 
                    # is_valid is called with the updated set? 
                    # No, is_valid(list(edges) + [e]) checks the hypothetical set.
                    # If we add it, we should continue checking other candidates against the NEW set?
                    # That's expensive. 
                    # Let's just do one pass, then loop.
        return list(edges)

    # Main Loop
    start_time = time.time()
    max_time = 50 # Leave 10s buffer
    
    for i in range(1000): # Max iterations
        if time.time() - start_time > max_time:
            break
        
        # Initialize
        current_edges = all_edges[:]
        random.shuffle(current_edges)
        
        # Repair
        current_edges = repair(current_edges)
        
        # Local Search
        current_edges = local_search_add(current_edges)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
        # If we reached a very high count, we might stop early?
        # Max edges in K15 is 105.
        # If best_count is high, we can try to improve further with swaps?
        # For now, just restart.

    return [list(e) for e in best_edges]
