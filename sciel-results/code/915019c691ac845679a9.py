import random
import time

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing the number of edges.
    """
    n = 20
    deadline = time.time() + 110.0  # Safety margin
    
    best_edges = 0
    best_edge_list = []
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to check validity and count common neighbors
    # neigh is a list of integers where bit i is set if vertex i is a neighbor
    
    def popcount(x):
        return x.bit_count()
    
    def is_valid(neigh):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(neigh[u] & neigh[v]) >= 3:
                    return False
        return True

    def repair(neigh, edges_set):
        """
        Remove edges to make the graph K2,3-free.
        Returns the modified neigh and edges_set.
        """
        # We loop until valid. 
        # To be efficient, we pick a violating pair and remove an edge.
        # We limit iterations to prevent infinite loops if logic is flawed (though it shouldn't be).
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline:
                return neigh, edges_set
            
            # Find a violating pair
            violated = False
            for u in range(n):
                for v in range(u + 1, n):
                    common = neigh[u] & neigh[v]
                    if popcount(common) >= 3:
                        violated = True
                        # Pick a random common neighbor to remove edge to
                        # Get list of set bits in common
                        common_list = [w for w in range(n) if (common >> w) & 1]
                        w = random.choice(common_list)
                        
                        # Remove edge (u, w) or (v, w). 
                        # Removing (u, w) affects pairs (u, x) for all x in neigh[u].
                        # Let's just remove (u, w) for simplicity.
                        # Actually, removing (u, w) reduces deg(u).
                        # We must ensure we don't remove a non-existent edge.
                        # (u, w) is an edge because w is in neigh[u].
                        
                        # Remove edge (u, w)
                        edges_set.discard((u, w))
                        edges_set.discard((w, u)) # Just in case, though we store u<v
                        
                        # Update neigh
                        neigh[u] &= ~(1 << w)
                        neigh[w] &= ~(1 << u)
                        
                        break
                if violated:
                    break
            
            if not violated:
                break
        
        return neigh, edges_set

    def improve(neigh, edges_set):
        """
        Try to add missing edges greedily.
        """
        # List of missing edges
        missing_edges = [e for e in all_edges if e not in edges_set]
        
        # We try to add them in random order
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            
            # Check if adding (u, v) creates a K2,3
            # New common neighbors of u and v would be:
            # Currently common = neigh[u] & neigh[v]
            # If we add (u, v), v becomes a neighbor of u, and u becomes a neighbor of v.
            # Does this create a K2,3?
            # A K2,3 involves two vertices having 3 common neighbors.
            # Adding (u, v) increases the degree of u and v.
            # It might create new common neighbors for pairs (u, x) and (v, x).
            
            # Let's check all pairs involving u or v.
            # Actually, the condition is: for all pairs (a, b), popcount(neigh[a] & neigh[b]) < 3.
            # Adding (u, v) changes:
            # 1. neigh[u] gets bit v.
            # 2. neigh[v] gets bit u.
            
            # This affects common neighbor counts for:
            # - Pair (u, v): Common neighbors are (neigh[u] & neigh[v]). 
            #   Note: u is not in neigh[u], v is not in neigh[v].
            #   So common neighbors of u and v are vertices w such that w~u and w~v.
            #   Adding edge (u,v) does NOT change the set of common neighbors of u and v.
            #   Wait, if w~u and w~v, w is a common neighbor.
            #   The edge (u,v) itself is not a "common neighbor".
            #   So popcount(neigh[u] & neigh[v]) remains the same.
            
            # - Pairs (u, w) where w != v:
            #   Common neighbors of u and w: vertices x such that x~u and x~w.
            #   Previously, v was NOT a neighbor of u.
            #   Now v IS a neighbor of u.
            #   If v~w, then v becomes a NEW common neighbor of u and w.
            #   So we must check if v is a neighbor of w.
            #   If v~w, then common count for (u, w) increases by 1.
            #   If it was already 2, it becomes 3 -> Invalid.
            
            # - Pairs (v, w) where w != u:
            #   Similarly, u becomes a new neighbor of v.
            #   If u~w, then u becomes a NEW common neighbor of v and w.
            #   Check if u~w. If so, common count for (v, w) increases by 1.
            
            # So, adding (u, v) is valid if:
            # 1. popcount(neigh[u] & neigh[v]) < 3 (Always true if graph was valid before? 
            #    Yes, because adding edge doesn't change common neighbors of u,v).
            #    Wait, is it possible that adding (u,v) makes (u,v) a common neighbor of some pair?
            #    No, u and v are distinct vertices.
            
            # 2. For all w in neigh[v] (w != u):
            #    If w is connected to u, then v is a common neighbor of u and w.
            #    Actually, we need to check if adding (u,v) causes any pair to reach 3 common neighbors.
            #    The pairs affected are (u, w) for w in neigh[v] and (v, w) for w in neigh[u].
            
            valid_add = True
            
            # Check pairs (u, w) for w in neigh[v]
            # Note: w is in neigh[v]. We are adding (u,v).
            # If w is also in neigh[u], then w is a common neighbor of u and v.
            # But we are checking common neighbors of (u, w).
            # Common neighbors of u and w:
            # New candidate: v.
            # v is a neighbor of u (new).
            # v is a neighbor of w (since w in neigh[v]).
            # So v is a common neighbor of u and w.
            # Before adding (u,v), was v a common neighbor of u and w?
            # No, because v was not a neighbor of u.
            # So the count increases by 1.
            # We need to ensure the OLD count was < 2.
            # Old count = popcount(neigh[u] & neigh[w]).
            # If popcount(neigh[u] & neigh[w]) >= 2, then adding (u,v) makes it 3. Invalid.
            
            for w in range(n):
                if w == u or w == v:
                    continue
                if (neigh[v] >> w) & 1:
                    # w is neighbor of v
                    # Check if w is neighbor of u
                    if (neigh[u] >> w) & 1:
                        # w is common neighbor of u and v
                        # This doesn't directly affect the validity of adding (u,v) for pair (u,v)
                        # But we need to check pair (u, w) and (v, w)
                        pass
                
                # Check pair (u, w)
                # If w is neighbor of v, then v becomes common neighbor of u and w.
                # We need old common count of u and w to be < 2.
                if (neigh[v] >> w) & 1:
                    old_common_uw = popcount(neigh[u] & neigh[w])
                    if old_common_uw >= 2:
                        valid_add = False
                        break
            
            if not valid_add:
                continue
                
            # Check pairs (v, w) for w in neigh[u]
            for w in range(n):
                if w == u or w == v:
                    continue
                if (neigh[u] >> w) & 1:
                    # w is neighbor of u
                    # u becomes common neighbor of v and w.
                    old_common_vw = popcount(neigh[v] & neigh[w])
                    if old_common_vw >= 2:
                        valid_add = False
                        break
            
            if valid_add:
                # Add edge (u, v)
                edges_set.add((u, v))
                neigh[u] |= (1 << v)
                neigh[v] |= (1 << u)
        
        return neigh, edges_set

    # Main Loop
    for restart in range(10000):
        if time.time() > deadline:
            break
        
        # Initialize random graph
        # Probability 0.5 to 0.7
        p = random.uniform(0.4, 0.7)
        neigh = [0] * n
        edges_set = set()
        
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    edges_set.add((u, v))
                    neigh[u] |= (1 << v)
                    neigh[v] |= (1 << u)
        
        # Repair
        neigh, edges_set = repair(neigh, edges_set)
        
        # Improve
        neigh, edges_set = improve(neigh, edges_set)
        
        # Check if better
        current_edges = len(edges_set)
        if current_edges > best_edges:
            best_edges = current_edges
            best_edge_list = [list(e) for e in edges_set]
            
            # Optional: If we reach a very high number, we can stop early?
            # Theoretical max is around 50-60. If we hit 50, it's likely near optimal.
            # But let's let it run to find 51, 52 etc.
            
    return best_edge_list
