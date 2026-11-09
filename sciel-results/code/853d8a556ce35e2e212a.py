import itertools
import time
import random

def _is_non_2_colorable(n, edge_bitmasks):
    """
    Checks if the hypergraph is NOT 2-colorable using fast bitmask backtracking.
    Returns True if NOT 2-colorable (i.e., no proper 2-colouring exists).
    Returns False if 2-colorable.
    """
    # edge_bitmasks: list of ints, each int has 5 bits set corresponding to vertices.
    
    # colors: 0 for unassigned, 1 for color 0, 2 for color 1? 
    # Better: use two bitmasks: mask0 (vertices colored 0), mask1 (vertices colored 1).
    # Unassigned are ~(mask0 | mask1) & ((1 << n) - 1).
    
    full_mask = (1 << n) - 1
    
    def backtrack(mask0, mask1):
        # Check for conflicts and propagate
        # If any edge is subset of mask0 or subset of mask1, conflict.
        # If an edge has 4 bits in mask0 and 1 unassigned, that unassigned must be 1.
        
        # Optimization: Check edges
        # To speed up, we can pre-process edges? 
        # For now, iterate.
        
        changed = True
        while changed:
            changed = False
            for e in edge_bitmasks:
                # Count bits in e & mask0, e & mask1
                c0 = bin(e & mask0).count('1')
                c1 = bin(e & mask1).count('1')
                
                if c0 == 5 or c1 == 5:
                    return False # Conflict
                
                # Unit propagation
                # Unassigned bits in this edge
                unassigned_in_e = e & ~(mask0 | mask1)
                
                if c0 == 4 and unassigned_in_e:
                    # The single unassigned bit must be 1
                    # It must be the only bit in unassigned_in_e
                    if unassigned_in_e & (unassigned_in_e - 1) == 0:
                        # It's a power of 2
                        bit = unassigned_in_e
                        mask1 |= bit
                        changed = True
                    else:
                        # This case shouldn't happen if we check c0+c1+unassigned_count == 5
                        # But if c0=4, unassigned count must be 1.
                        pass
                elif c1 == 4 and unassigned_in_e:
                    if unassigned_in_e & (unassigned_in_e - 1) == 0:
                        bit = unassigned_in_e
                        mask0 |= bit
                        changed = True
        
        # Check if all assigned
        if (mask0 | mask1) == full_mask:
            return True # All assigned, no conflict found so far -> 2-colorable
        
        # Pick a variable to assign
        # Heuristic: Pick a vertex involved in the most constrained edges?
        # For speed, just pick the lowest unassigned bit.
        unassigned_all = full_mask & ~(mask0 | mask1)
        if unassigned_all == 0:
            return True
            
        # Get lowest bit
        v_bit = unassigned_all & -unassigned_all
        
        # Try color 0
        if backtrack(mask0 | v_bit, mask1):
            return True
        # Try color 1
        if backtrack(mask0, mask1 | v_bit):
            return True
            
        return False

    # Initial state
    if backtrack(0, 0):
        return False # It IS 2-colorable
    else:
        return True # It is NOT 2-colorable

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # We will try different n and m
    # Strategy: Focus on small n to make checking fast.
    # n=10, 11, 12, 13, 14, 15
    
    # Pre-generate all possible edges for small n?
    # For n=12, C(12,5)=792.
    # For n=15, C(15,5)=3003.
    
    # Let's try n=12 first. If we find a non-2-colorable one, it's likely small.
    # If not, increase n.
    
    for n in range(10, 16):
        if time.time() > deadline:
            break
            
        all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        # Convert to bitmasks for the checker
        all_edge_bitmasks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            all_edge_bitmasks.append(mask)
        
        # Shuffle
        random.shuffle(all_edges)
        random.shuffle(all_edge_bitmasks)
        
        # Try to build a hypergraph by adding edges
        # Start with a random subset
        # We want to minimize edges, so we start small and add until non-2-colorable?
        # Or start large and remove?
        # Removing is safer to keep non-2-colorability.
        # But starting large requires checking a large graph first.
        
        # Let's try: Pick a random set of m edges. Check.
        # If non-2-colorable, try to reduce m.
        
        # To increase probability of non-2-colorability, we can bias towards
        # edges that share many vertices.
        
        # Simple random search
        max_m = min(len(all_edges), 60) # Cap to keep checking fast
        
        # We'll do multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Random m between 30 and max_m
            m = random.randint(30, max_m)
            
            # Select m random edges
            indices = random.sample(range(len(all_edges)), m)
            current_edges = [all_edges[i] for i in indices]
            current_bitmasks = [all_edge_bitmasks[i] for i in indices]
            
            # Check
            if _is_non_2_colorable(n, current_bitmasks):
                # Found a non-2-colorable hypergraph!
                # Try to minimize
                current_set = set(tuple(e) for e in current_edges)
                current_mask_set = set(current_bitmasks)
                
                # Greedy removal
                # Shuffle current edges
                rem_list = list(current_edges)
                random.shuffle(rem_list)
                
                for e in rem_list:
                    if time.time() > deadline:
                        break
                    if len(current_set) <= best_count:
                        break # Can't improve
                    # Try removing e
                    if e in current_set:
                        current_set.remove(e)
                        # Rebuild bitmasks? Or just check the set?
                        # The checker takes a list of bitmasks.
                        # We need to map edges to bitmasks.
                        # Since n is small, we can just convert the remaining set to bitmasks.
                        # This is O(m * 5). Fast enough.
                        
                        remaining_edges = list(current_set)
                        remaining_bitmasks = []
                        for re in remaining_edges:
                            mask = 0
                            for v in re:
                                mask |= (1 << v)
                            remaining_bitmasks.append(mask)
                        
                        if _is_non_2_colorable(n, remaining_bitmasks):
                            # Keep removal
                            best_count = len(current_set)
                            best_edges = [list(e) for e in current_set]
                            best_n = n
                        else:
                            # Restore
                            current_set.add(e)
                
                # If we improved, continue searching for better
                if best_count <= 51:
                    # We hit the known best, we can stop early if we want, 
                    # but let's see if we can go lower? Unlikely.
                    pass
                    
    if best_edges is None:
        # Fallback: Return a known valid one? 
        # If we didn't find one, we are in trouble.
        # But with fast checker, we should have found one for n=12 or 13.
        # If not, return a dummy valid one? No, dummy might be invalid.
        # Let's force a specific known non-2-colorable structure if search fails.
        # One simple non-2-colorable 5-uniform hypergraph:
        # Take K_5^5 (1 edge) -> 2-colorable.
        # Take K_6^5 (6 edges) -> 2-colorable? 
        # If we take all 5-edges of K_6, is it 2-colorable?
        # Vertices 0-5. Edges are all subsets of size 5.
        # Edge {0,1,2,3,4}, {0,1,2,3,5}, etc.
        # If 0,1,2,3 are 0, then 4 must be 1, 5 must be 1.
        # Edge {0,1,2,4,5}: 0,0,0,1,1. OK.
        # It seems K_6^5 is 2-colorable.
        
        # Let's just return the best found, or an empty set if none (score 0).
        # But the search should find one.
        
        # As a last resort, return a large dense graph which is likely non-2-colorable
        # but we need to verify it.
        n = 15
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        # This has 3003 edges. Limit is 200.
        # So we can't return this.
        
        # We must have found something in the loop.
        # If not, we return an invalid structure to avoid crash, but score 0.
        return {'n': 5, 'edges': [[0,1,2,3,4]]}
        
    return {'n': best_n, 'edges': best_edges}
