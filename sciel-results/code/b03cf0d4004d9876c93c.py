import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # We know graphs with 20 vertices exist.
    # We aim to find one.
    
    def is_regular_subset(mask, n, adj):
        # mask is an integer bitmask of the subset
        # adj is a list of int bitmasks for each vertex
        # Returns True if induced subgraph is regular
        # First, check if it's empty or size < 6? 
        # The caller ensures size >= 6.
        
        # Get degrees
        # To be fast, we can iterate through set bits
        # But for n=20, iterating 0..19 is fine.
        degs = []
        m = mask
        # Optimization: if any vertex in mask has 0 or (size-1) degree, 
        # it's not regular unless size is 1 or 2 (not our case) 
        # Actually, regular means all degrees equal.
        
        # Let's just compute degrees for all v in 0..n-1 if v is in mask
        # This is O(n) per subset.
        # 10^6 subsets * 20 ops = 20M ops. Might be slow in Python (1-2s).
        # Acceptable for a few checks.
        
        # Better: Precompute? No, adj changes.
        
        # Let's iterate
        v = 0
        while v < n:
            if (mask >> v) & 1:
                d = (adj[v] & mask).bit_count()
                degs.append(d)
            v += 1
        
        if not degs:
            return False
        first = degs[0]
        for d in degs[1:]:
            if d != first:
                return False
        return True

    def count_bad_subsets(n, adj):
        bad_count = 0
        # Iterate all masks with bit_count >= 6
        # There are 2^20 masks.
        # We can iterate 0 to 2^n - 1 and check bit_count.
        # 1M iterations is fast.
        for mask in range(1 << n):
            if mask.bit_count() >= 6:
                if is_regular_subset(mask, n, adj):
                    bad_count += 1
        return bad_count

    # Start with n=20
    n = 20
    
    # Try a few random graphs
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Random graph
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        bad_count = count_bad_subsets(n, adj)
        
        if bad_count == 0:
            return {'n': n, 'edges': edges}
        
        # Greedy Repair
        # Find all bad subsets
        bad_masks = []
        for mask in range(1 << n):
            if mask.bit_count() >= 6:
                if is_regular_subset(mask, n, adj):
                    bad_masks.append(mask)
        
        if not bad_masks:
             return {'n': n, 'edges': edges}

        # Try to fix by flipping edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Pick a random bad mask
            # Or iterate through them?
            # Let's pick one random bad mask
            mask = random.choice(bad_masks)
            
            # Get vertices in mask
            verts = [v for v in range(n) if (mask >> v) & 1]
            
            # Try flipping an edge within verts
            # Randomly permute pairs
            pairs = [(u, v) for i, u in enumerate(verts) for v in verts[i+1:]]
            random.shuffle(pairs)
            
            fixed = False
            for u, v in pairs:
                # Flip edge (u, v)
                # Check if it reduces bad count
                # We don't want to recompute full count.
                # Heuristic: Just flip and check if this specific mask is still bad?
                # If we fix this mask, it might break others.
                # Let's just flip and recompute bad_count.
                # But recompute is expensive.
                
                # Alternative: Just try flipping and see if bad_count decreases.
                # We need a faster delta.
                
                # Let's stick to simple: Flip, recompute bad_count.
                # If it takes 1s to recompute, and we have 100s budget, we can do ~100 steps.
                # This is likely to fail for n=20.
                
                # Better: Use the structure.
                # If mask is regular, all degrees are d.
                # Flipping (u,v) changes deg(u) and deg(v) by +/- 1.
                # So mask becomes non-regular if d != d+1 or d != d-1 (always true for d>0 or d<size-1).
                # Wait. If we flip an edge (u,v) inside a regular set S:
                # deg(u) becomes d+1, deg(v) becomes d+1. Others d.
                # So S is no longer regular (unless d+1 == d, impossible).
                # So flipping ANY edge inside a regular set S makes S non-regular.
                # However, it might create new regular sets or make other regular sets non-regular.
                
                # So, if we flip an edge (u,v) contained in a bad set S, S is fixed.
                # We should check if this flip creates new bad sets.
                
                # Let's try flipping and checking the global count.
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                if (u, v) in [(e[0], e[1]) for e in edges]:
                    # Remove from edges
                    # This is O(E).
                    pass 
                # Actually, maintaining edges list is annoying.
                # Let's just reconstruct edges at the end.
                
                new_bad_count = count_bad_subsets(n, adj)
                
                if new_bad_count < bad_count:
                    bad_count = new_bad_count
                    improved = True
                    fixed = True
                    # Update bad_masks?
                    # Just recompute bad_masks in next loop iteration?
                    # Or break and recompute.
                    break
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            
            if fixed:
                # Recompute bad_masks
                bad_masks = []
                for mask in range(1 << n):
                    if mask.bit_count() >= 6:
                        if is_regular_subset(mask, n, adj):
                            bad_masks.append(mask)
                if not bad_masks:
                    # Success
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
            else:
                # No single edge flip improved the count
                # Try a random restart?
                # Or accept a worse state (Simulated Annealing)?
                # For now, let's just restart the whole graph if stuck.
                break

    # If we failed to find n=20, try n=19, 18...
    # But the task says 20 is known.
    # If we run out of time, return the best valid graph found.
    # We haven't tracked best valid graph.
    # Let's initialize with a known small valid graph (n=5 empty).
    best_graph = {'n': 5, 'edges': []}
    
    # If we found a valid graph in the loop, we returned.
    # If not, we return the fallback.
    # To be safe, let's try n=6, 7... with random search?
    # No, just return n=5 if 20 fails.
    # But we can do better.
    # Let's try n=6.
    # Random graph on 6 vertices.
    # Check if it has no regular subset of size >= 6.
    # Only subset is the whole graph.
    # So we just need a non-regular graph on 6 vertices.
    # Easy.
    
    # Let's just return n=5 to be safe if 20 fails.
    # But we should try harder for 20.
    
    return best_graph
