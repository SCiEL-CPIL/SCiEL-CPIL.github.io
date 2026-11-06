import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # We want to maximize n.
    # Try n=47, then 46, etc.
    # The known bound is R(C4, K1,39) > 46 or 47.
    # If we find a valid graph for n=47, score is 47/39.
    # If we find for n=46, score is 46/39.
    # Baseline is n=39 (score 1.0).
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg_required = n - 39
        
        # If min_deg_required <= 0, empty graph works (n=39)
        if min_deg_required <= 0:
            return {'n': n, 'edges': []}
            
        # Try to construct for this n
        result = try_construct_additive(n, min_deg_required, deadline)
        if result:
            return result
            
    # Fallback
    return {'n': 39, 'edges': []}

def try_construct_additive(n, min_deg_required, deadline):
    """
    Greedy edge addition.
    Start with empty graph.
    Repeatedly add an edge (u, v) such that:
    1. deg(u) < min_deg_required OR deg(v) < min_deg_required (to fix degrees)
       Actually, we must ensure ALL degrees reach min_deg_required.
       So we should prioritize vertices with low degree.
    2. Adding (u, v) does not create a C4.
       This is equivalent to: |N(u) ∩ N(v)| == 0.
       
    We will run multiple random restarts with different randomizations.
    """
    
    max_restarts = 5000
    restart_count = 0
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # adj[i] is bitmask of neighbors
        adj = [0] * n
        deg = [0] * n
        
        # We need to add edges until all deg[i] >= min_deg_required
        # Total edges needed is at least n * min_deg_required / 2
        
        # Strategy:
        # Maintain a list of "deficient" vertices (deg < min_deg_required).
        # Pick a deficient vertex u.
        # Pick a candidate v such that:
        #   - v is not u
        #   - (u,v) not in adj
        #   - N(u) ∩ N(v) is empty (no C4 created)
        #   - Prefer v that is also deficient? Or just any valid v?
        #     If we prefer deficient v, we fix two at once.
        
        success = False
        stuck = False
        
        # To avoid getting stuck in local optima, we can randomize the order of deficient vertices
        # or the choice of v.
        
        while time.time() < deadline:
            # Check if we are done
            if min(deg) >= min_deg_required:
                # Verify no C4 (should be true by construction, but safe to check? 
                # No, construction guarantees it. 
                # However, we should verify min degree again just in case of logic errors)
                if min(deg) >= min_deg_required:
                    # Extract edges
                    edges = []
                    for i in range(n):
                        mask = adj[i]
                        while mask:
                            j = (mask & -mask).bit_length() - 1
                            edges.append([i, j])
                            mask &= mask - 1
                    # Deduplicate? The loop `for i` adds (i,j) for all j in N(i).
                    # Since graph is undirected, (i,j) and (j,i) are both added.
                    # The interface expects a list of edges. Usually unique pairs.
                    # Let's construct unique pairs.
                    edges = []
                    for i in range(n):
                        mask = adj[i]
                        while mask:
                            j = (mask & -mask).bit_length() - 1
                            if i < j:
                                edges.append([i, j])
                            mask &= mask - 1
                    return {'n': n, 'edges': edges}
                
            # Find a deficient vertex
            # To add randomness, pick a random deficient vertex
            deficient = [i for i in range(n) if deg[i] < min_deg_required]
            if not deficient:
                # Should have returned above
                break
                
            u = random.choice(deficient)
            
            # Find valid candidates v
            # v must be:
            # 1. v != u
            # 2. not adjacent to u
            # 3. N(u) ∩ N(v) == 0
            
            # Candidates are all vertices except u and current neighbors of u
            # Let's iterate through all v and check
            
            best_v = -1
            best_score = -1
            candidates = []
            
            # Optimization: Pre-compute N(u)
            nu_mask = adj[u]
            
            for v in range(n):
                if v == u:
                    continue
                if nu_mask & (1 << v):
                    continue # Already connected
                
                # Check C4 condition: N(u) ∩ N(v) must be empty
                # If N(u) is empty, intersection is empty.
                # If N(u) is not empty, check intersection with N(v)
                if nu_mask & adj[v]:
                    continue
                
                # Valid candidate
                # Score: prefer v that is deficient?
                score = 0
                if deg[v] < min_deg_required:
                    score += 10
                # Add some randomness
                score += random.random()
                
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                # No valid neighbor for u. Stuck.
                stuck = True
                break
            
            # Add edge (u, best_v)
            v = best_v
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
            # If we get stuck, we fail this restart
            
        # End while loop for this restart
        
    return None

