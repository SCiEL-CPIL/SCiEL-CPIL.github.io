import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin
    
    # Initialize with complete graph K_20
    # adj[u] is bitmask of neighbors
    adj = [0] * n
    for u in range(n):
        for v in range(n):
            if u != v:
                adj[u] |= (1 << v)
                
    # Calculate initial common neighbor counts
    common = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            # Count bits in adj[u] & adj[v]
            # Since it's K20, adj[u] & adj[v] is all bits except u and v
            # So count is n - 2 = 18
            common[u][v] = n - 2
            common[v][u] = n - 2
            
    # Current state: all edges present
    # removed_edges is a set of (u, v) tuples where u < v
    removed_edges = set()
    
    def calc_violations(common):
        v_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if common[u][v] > 2:
                    v_count += common[u][v] - 2
        return v_count

    current_violations = calc_violations(common)
    # Initial cost: |R| + M * V
    # M should be large enough to prioritize V=0
    M = 1000 
    current_cost = len(removed_edges) + M * current_violations
    
    best_cost = current_cost
    best_removed = set(removed_edges)
    best_edges_list = []
    
    # Generate list of all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute neighbors for fast iteration?
    # We will use bit manipulation for adding/removing edges
    
    def add_edge_to_removed(u, v):
        """Add (u,v) to removed set (remove from graph). Update common counts."""
        nonlocal current_violations, current_cost
        if u > v:
            u, v = v, u
        if (u, v) in removed_edges:
            return False
        
        removed_edges.add((u, v))
        
        # Remove edge (u,v) from graph
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update common counts
        # For each w in N(v) \ {u}, common[u][w] decreases by 1
        # Note: N(v) is adj[v] AFTER removal? 
        # The common neighbors of u and w are affected if w is a neighbor of v.
        # If w is in N(v) and w != u, then v is a common neighbor of u and w?
        # No, v is a neighbor of w (since w in N(v)) and v is a neighbor of u (edge uv existed).
        # So v was a common neighbor of u and w.
        # Removing (u,v) means v is no longer a neighbor of u.
        # So v is no longer a common neighbor of u and w.
        # Thus common[u][w] decreases by 1.
        
        # We need to iterate over neighbors of v (before removal? or after?)
        # The common neighbors of u and w that are lost are exactly those x such that x is neighbor of u and w.
        # Specifically, v was such an x.
        # Are there others? No, only the edge (u,v) is removed.
        # So for each w in N(v) \ {u}, common[u][w] -= 1.
        # And for each w in N(u) \ {v}, common[v][w] -= 1.
        
        # We must use the adjacency BEFORE removal to find the neighbors that were connected.
        # But we already updated adj[u] and adj[v].
        # Let's store old neighbors or just use the fact that we know which neighbors are affected.
        # Actually, the set of w such that w in N(v) is adj[v] BEFORE removal.
        # Let's fetch it before updating adj.
        
        # REVERT for a moment in logic:
        # We should compute the sets to update BEFORE modifying adj.
        
        # Let's refactor the update logic to be cleaner.
        
        return True

    # Let's rewrite the update logic properly inside the SA loop to avoid state confusion.
    
    def get_neighbors(mask):
        """Yield indices of set bits in mask."""
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask -= lsb

    # Re-initialize state for SA
    removed_edges = set()
    adj = [0] * n
    for u in range(n):
        for v in range(n):
            if u != v:
                adj[u] |= (1 << v)
                
    common = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            common[u][v] = n - 2
            common[v][u] = n - 2
            
    current_violations = calc_violations(common)
    current_cost = len(removed_edges) + M * current_violations
    best_cost = current_cost
    best_removed = set(removed_edges)
    
    # SA Loop
    while time.time() - start_time < max_time and T > min_T:
        # Pick a random edge to toggle
        u, v = random.choice(all_edges)
        
        # Calculate delta cost
        # If (u,v) is in removed_edges, we are ADDING it to graph.
        # If (u,v) is NOT in removed_edges, we are REMOVING it from graph.
        
        if (u, v) in removed_edges:
            # Move: Add edge to graph (remove from removed set)
            # Check if it's valid to add? No, SA allows invalid.
            # We need to calculate new violations.
            
            # Delta in violations:
            # Adding (u,v) increases common[u][w] for w in N(v)\{u}
            # and common[v][w] for w in N(u)\{v}
            
            delta_v = 0
            # Neighbors of v (excluding u)
            mask_v = adj[v] & ~(1 << u)
            for w in get_neighbors(mask_v):
                old_c = common[u][w]
                new_c = old_c + 1
                # Contribution to violation: max(0, c - 2)
                old_contrib = max(0, old_c - 2)
                new_contrib = max(0, new_c - 2)
                delta_v += new_contrib - old_contrib
                
            # Neighbors of u (excluding v)
            mask_u = adj[u] & ~(1 << v)
            for w in get_neighbors(mask_u):
                old_c = common[v][w]
                new_c = old_c + 1
                old_contrib = max(0, old_c - 2)
                new_contrib = max(0, new_c - 2)
                delta_v += new_contrib - old_contrib
                
            delta_r = -1  # One less removed edge
            delta_cost = delta_r + M * delta_v
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept move
                removed_edges.remove((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Update common
                for w in get_neighbors(mask_v):
                    common[u][w] += 1
                    common[w][u] += 1
                for w in get_neighbors(mask_u):
                    common[v][w] += 1
                    common[w][v] += 1
                    
                current_violations += delta_v
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_removed = set(removed_edges)
                    
        else:
            # Move: Remove edge from graph (add to removed set)
            # This always decreases or keeps violations same.
            
            delta_v = 0
            mask_v = adj[v] & ~(1 << u)
            for w in get_neighbors(mask_v):
                old_c = common[u][w]
                new_c = old_c - 1
                old_contrib = max(0, old_c - 2)
                new_contrib = max(0, new_c - 2)
                delta_v += new_contrib - old_contrib
                
            mask_u = adj[u] & ~(1 << v)
            for w in get_neighbors(mask_u):
                old_c = common[v][w]
                new_c = old_c - 1
                old_contrib = max(0, old_c - 2)
                new_contrib = max(0, new_c - 2)
                delta_v += new_contrib - old_contrib
                
            delta_r = 1
            delta_cost = delta_r + M * delta_v
            
            # Since M is large and delta_v <= 0, delta_cost is likely negative or small positive.
            # If delta_v is negative enough, it's a good move.
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept move
                removed_edges.add((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Update common
                for w in get_neighbors(mask_v):
                    common[u][w] -= 1
                    common[w][u] -= 1
                for w in get_neighbors(mask_u):
                    common[v][w] -= 1
                    common[w][v] -= 1
                    
                current_violations += delta_v
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_removed = set(removed_edges)
                    
        T *= cooling_rate

    # Construct final edge list from best_removed
    final_edges = []
    for u, v in all_edges:
        if (u, v) not in best_removed:
            final_edges.append([u, v])
            
    return final_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
