import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    def get_edges(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_masks[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def random_adj(n, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def precompute_masks(n, k):
        masks = []
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            masks.append(mask)
        return masks

    # Precompute masks for size 6 for n=20 and n=21
    # We will search for n=21 first. If it fails, we keep the best n=20 candidate.
    # But we need to be efficient.
    
    best_result = {'n': 5, 'edges': []}
    
    # We'll try to find a valid graph for n=21.
    # If we can't, we fall back to n=20.
    # Actually, let's just run the search for a target n.
    
    def find_valid_graph(target_n, time_limit):
        # Precompute masks for size 6
        masks_6 = precompute_masks(target_n, 6)
        
        # Also precompute masks for sizes 7..target_n for final check?
        # No, do it on the fly for final check to save memory/init time, 
        # or precompute if fast. C(21,7)=116k, C(21,8)=203k... 
        # Total > 1M. Precomputing all is ~1M integers. ~8MB. Acceptable.
        # But iterating 1M masks in Python is slow.
        # We will only do the full check when size-6 check passes.
        
        # Search Loop
        # We will try multiple random restarts.
        
        for seed in range(100): # Max 100 restarts, but time limited
            if elapsed() > time_limit:
                break
                
            adj = random_adj(target_n, seed)
            
            # Local Search
            steps = 0
            max_steps = 5000
            while steps < max_steps:
                if elapsed() > time_limit:
                    return None, adj # Return current adj for potential fallback? No, just None.
                
                # Check for size-6 regular subgraph
                viol_mask = None
                for mask in masks_6:
                    # Check if induced subgraph on mask is regular
                    # Get vertices in mask? We need to iterate bits.
                    # Optimization: 
                    # For each v in range(target_n):
                    #   if mask & (1<<v):
                    #      deg = (adj[v] & mask).bit_count()
                    #      if first: first_deg = deg
                    #      elif deg != first_deg: break
                    # If no break, it's regular.
                    
                    first_deg = -1
                    is_reg = True
                    for v in range(target_n):
                        if mask & (1 << v):
                            deg = (adj[v] & mask).bit_count()
                            if first_deg == -1:
                                first_deg = deg
                            elif deg != first_deg:
                                is_reg = False
                                break
                    if is_reg:
                        viol_mask = mask
                        break
                
                if viol_mask is None:
                    # No size-6 regular subgraph found.
                    # Perform FULL check (sizes 6 to target_n)
                    valid = True
                    for k in range(6, target_n + 1):
                        if k == 6:
                            continue # Already checked
                        # Generate combinations for k
                        # This is expensive. 
                        # Optimization: If we found a valid graph for n=20, we know it exists.
                        # For n=21, if size-6 is clean, it's likely valid.
                        # But we must be sure.
                        # Let's do a quick check for k=7..target_n.
                        for subset in itertools.combinations(range(target_n), k):
                            mask = 0
                            for v in subset:
                                mask |= (1 << v)
                            first_deg = -1
                            is_reg = True
                            for v in range(target_n):
                                if mask & (1 << v):
                                    deg = (adj[v] & mask).bit_count()
                                    if first_deg == -1:
                                        first_deg = deg
                                    elif deg != first_deg:
                                        is_reg = False
                                        break
                            if is_reg:
                                valid = False
                                # Try to fix this specific violation?
                                # For now, just mark invalid and break.
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return adj, adj # Found valid graph
                
                # If violation found (size 6 or larger in full check), fix it
                if viol_mask is not None:
                    # Flip a random edge within the violating set
                    vertices_in_mask = [v for v in range(target_n) if viol_mask & (1 << v)]
                    if len(vertices_in_mask) >= 2:
                        u, v = random.sample(vertices_in_mask, 2)
                        # Flip edge u-v
                        if adj[u] & (1 << v):
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                else:
                    # This case implies we found a larger violation in the full check but viol_mask was None?
                    # No, if viol_mask is None, size-6 was clean.
                    # If full check failed, we didn't store the violating mask.
                    # Let's restart the local search or just break.
                    # Actually, if size-6 is clean but size-7 is not, flipping a random edge in the whole graph might help.
                    # But this is rare. Let's just break and try next seed.
                    break
                
                steps += 1
        
        return None, None

    # Try n=21
    target_n = 21
    adj, _ = find_valid_graph(target_n, DEADLINE - 10)
    if adj is not None:
        return {'n': target_n, 'edges': get_edges(adj, target_n)}
        
    # If n=21 failed, try n=20
    # But we might have spent too much time.
    if elapsed() < DEADLINE - 20:
        target_n = 20
        adj, _ = find_valid_graph(target_n, DEADLINE - 5)
        if adj is not None:
            return {'n': target_n, 'edges': get_edges(adj, target_n)}
            
    # Fallback: Return a small valid graph? 
    # The task says "Maximize n". If we can't find 20 or 21, we should return the best we have.
    # But we need a valid graph.
    # A random graph on n=10 is likely valid?
    # Let's just generate a random graph on n=10 and check it fully.
    # Or better, return a known small valid graph? 
    # n=1: valid. n=2: valid.
    # Let's try to find a valid graph on n=10 quickly.
    
    # Actually, the prompt implies we should find a large one. 
    # If we time out, we should return the best *valid* graph found.
    # We didn't store intermediate valid graphs (only size-6 valid).
    # Let's modify find_valid_graph to return the last state if it was size-6 clean, 
    # and we verify it at the end? No, that's risky.
    
    # Let's just return a valid graph on n=6? 
    # Wait, "No set of 6 or more...". 
    # A graph on n=5 is always valid (no subset of size 6).
    # So n=5 is always a valid answer.
    # We can return n=5 if nothing else works.
    
    return {'n': 5, 'edges': []}
