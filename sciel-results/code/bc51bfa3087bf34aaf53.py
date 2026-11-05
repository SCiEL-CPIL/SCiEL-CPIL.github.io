"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 39
    best_edges = []
    
    # We aim for n=47, then 46
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
            
        min_deg_req = target_n - 39
        
        # If min_deg_req is 0 or negative, it's trivial (empty graph or any graph)
        if min_deg_req <= 0:
            return {'n': target_n, 'edges': []}
            
        found = False
        # Try multiple random restarts
        # For n=47, min_deg=8. For n=46, min_deg=7.
        max_restarts = 2000 if target_n == 46 else 5000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart)
            
            # Bitmasks for adjacency
            # Python integers are arbitrary precision, so this is efficient for n <= 64
            adj = [0] * target_n
            degrees = [0] * target_n
            
            # Helper to check if adding edge (u, v) is C4-free
            # A C4 is formed if |N(u) ∩ N(v)| >= 2
            # We maintain adj as bitmasks.
            
            # Greedy construction
            # We want to increase degrees of all vertices to at least min_deg_req
            # Strategy: Always pick the vertex with the currently lowest degree.
            # If it's below min_deg_req, try to add an edge.
            
            # To make it stochastic, we can shuffle the order of checking potential neighbors
            # or just rely on the vertex selection order.
            
            stuck_count = 0
            max_iterations = 10000
            
            # Pre-compute list of vertices for efficient lowest degree finding?
            # For n=47, a linear scan is fine.
            
            for _ in range(max_iterations):
                # Find vertex with minimum degree
                min_deg = min(degrees)
                if min_deg >= min_deg_req:
                    found = True
                    break
                
                # Find a vertex with this minimum degree
                # Randomize selection among ties to escape local minima
                candidates = [i for i in range(target_n) if degrees[i] == min_deg]
                v = random.choice(candidates)
                
                # Try to add an edge to v
                # Potential neighbors are those not already connected
                # We iterate through all non-neighbors
                added = False
                # Get list of non-neighbors
                # adj[v] is a bitmask of neighbors
                # We want u such that (adj[v] >> u) & 1 == 0
                # And (adj[v] & adj[u]).bit_count() < 2
                
                # To speed up, we can generate a random order of potential neighbors
                # Or just iterate 0..n-1
                # Since n is small, iteration is fast.
                
                # Shuffle potential neighbors?
                # Let's just iterate. If we get stuck, the restart handles it.
                # However, the order matters. Let's randomize the starting point or order.
                # Actually, just picking a random valid u is better.
                
                # Find all valid u
                valid_us = []
                for u in range(target_n):
                    if u == v:
                        continue
                    if (adj[v] >> u) & 1:
                        continue
                    # Check C4 condition
                    common = adj[v] & adj[u]
                    if common.bit_count() < 2:
                        valid_us.append(u)
                        
                if not valid_us:
                    # Stuck
                    stuck_count += 1
                    if stuck_count > 10:
                        break
                    # Try to remove an edge to unstick? 
                    # For simplicity, just break and restart.
                    break
                else:
                    # Pick a random valid u
                    u = random.choice(valid_us)
                    # Add edge
                    adj[v] |= (1 << u)
                    adj[u] |= (1 << v)
                    degrees[v] += 1
                    degrees[u] += 1
                    stuck_count = 0
            
            if found:
                # Construct edge list
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                
                # Verify C4-free and min degree (sanity check)
                # This is O(n^3) roughly, but n is small
                is_c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (adj[i] >> j) & 1:
                            common = adj[i] & adj[j]
                            if common.bit_count() >= 2:
                                is_c4_free = False
                                break
                    if not is_c4_free:
                        break
                
                if is_c4_free and min(degrees) >= min_deg_req:
                    best_n = target_n
                    best_edges = edges
                    return {'n': best_n, 'edges': best_edges}
        
        # If we didn't find it for this target_n, continue to next (smaller) target_n
        # Note: If n=47 failed, we try n=46.
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
