"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    Uses an optimized greedy heuristic: iteratively select the vertex (number)
    that blocks the most other candidates. 
    Optimized by pre-computing blocked candidates via sum inversion.
    """
    selected = []
    sums = set()
    available = set(range(1, N + 1))
    
    while available:
        best_v = None
        max_degree = -1
        best_blocked_set = set()
        
        # Evaluate each candidate in 'available'
        for v in available:
            # 1. Check if v is compatible with current 'selected'
            # New sums introduced by v: 2v and v+s for s in selected
            new_sums_v = {2 * v}
            for s in selected:
                new_sums_v.add(v + s)
            
            # If any new sum collides with existing sums, v is invalid
            if sums.intersection(new_sums_v):
                continue
            
            # v is a valid candidate to add.
            # 2. Calculate 'degree': how many other available elements u
            #    become incompatible if we add v?
            
            # u is blocked if 2u is in new_sums_v OR (u+s) is in new_sums_v for any s in selected.
            # We generate the set of such u efficiently.
            blocked_candidates = set()
            
            for x in new_sums_v:
                # Case 1: 2u = x  => u = x/2
                if x % 2 == 0:
                    u_val = x // 2
                    if 1 <= u_val <= N:
                        blocked_candidates.add(u_val)
                
                # Case 2: u + s = x => u = x - s
                for s in selected:
                    u_val = x - s
                    if 1 <= u_val <= N:
                        blocked_candidates.add(u_val)
            
            # The blocked set is the intersection of blocked_candidates and available, excluding v
            common = blocked_candidates.intersection(available)
            if v in common:
                common.discard(v)
            
            degree = len(common)
            
            if degree > max_degree:
                max_degree = degree
                best_v = v
                best_blocked_set = common
        
        if best_v is None:
            # No valid vertex, set is maximal
            break
        
        # Update state with best_v
        selected.append(best_v)
        
        # Update sums
        sums.add(2 * best_v)
        for s in selected[:-1]:
            sums.add(best_v + s)
        
        # Update available: remove best_v and all u in best_blocked_set
        available.discard(best_v)
        available.difference_update(best_blocked_set)
        
    return selected
# EVOLVE-BLOCK-END
