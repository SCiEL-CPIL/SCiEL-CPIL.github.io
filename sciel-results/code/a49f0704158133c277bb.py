"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    Models the problem as finding a small maximal independent set in a conflict graph.
    Uses a greedy heuristic: iteratively select the vertex with the highest degree
    (blocks the most other candidates) to minimize the final set size.
    Optimized by calculating blocked vertices directly from the selected set.
    """
    selected = []
    sums = set()
    available = set(range(1, N + 1))
    
    while available:
        best_v = None
        max_degree = -1
        
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
            
            # u is blocked if adding v creates a sum collision for u.
            # Since u is in 'available', it is already compatible with 'selected'.
            # Thus, u is blocked iff it conflicts with the NEW sums introduced by v.
            # New sums: {2v} U {v+s | s in selected}
            
            # u conflicts with new sums if:
            # A) 2u is in new_sums_v
            # B) u+s is in new_sums_v for some s in selected U {v}
            
            # Let's derive the set of u that are blocked.
            # Instead of iterating all u, we generate candidate u's from the conditions.
            
            blocked_candidates = set()
            
            # Condition A: 2u in {2v} U {v+s}
            # 2u = 2v => u=v (ignore, u!=v)
            # 2u = v+s => u = (v+s)/2
            for s in selected:
                val = v + s
                if val % 2 == 0:
                    u = val // 2
                    if 1 <= u <= N:
                        blocked_candidates.add(u)
            
            # Condition B: u+s in {2v} U {v+s'} for s in selected, s' in selected U {v}
            
            # Case B1: u+s = 2v => u = 2v - s
            for s in selected:
                u = 2 * v - s
                if 1 <= u <= N:
                    blocked_candidates.add(u)
            
            # Case B2: u+s = v+s' => u = v + s' - s
            # s' can be v or any s2 in selected
            # If s' = v: u = v + v - s = 2v - s (Same as B1)
            # If s' = s2 in selected: u = v + s2 - s
            for s2 in selected:
                for s in selected:
                    u = v + s2 - s
                    if 1 <= u <= N:
                        blocked_candidates.add(u)
            
            # Also consider s' = v in the loop above?
            # My loop "for s2 in selected" covers s' in selected.
            # I handled s'=v in B1.
            # Wait, B2 is u+s = v+s'.
            # If s'=v, u+s = v+v => u = 2v-s. This is covered in B1.
            # If s' in selected, u = v+s'-s. This is covered in the nested loop.
            
            # We must also check if these u's are actually in 'available'.
            # And u != v.
            
            degree = 0
            for u in blocked_candidates:
                if u == v:
                    continue
                if u in available:
                    degree += 1
            
            if degree > max_degree:
                max_degree = degree
                best_v = v
        
        if best_v is None:
            # No valid vertex, set is maximal
            break
        
        # Update state with best_v
        selected.append(best_v)
        
        # Update sums
        sums.add(2 * best_v)
        for s in selected[:-1]:
            sums.add(best_v + s)
        
        # Update available: remove best_v and any u incompatible with new selected
        # We can reuse the logic: u is incompatible if it was blocked by best_v
        # OR if it was already incompatible (but available only contains compatible ones)
        # So available_new = available - {best_v} - (blocked_candidates for best_v)
        
        # Recalculate blocked for best_v to remove them
        blocked_by_best = set()
        # 2u = v+s
        for s in selected[:-1]: # selected[:-1] is the old selected
             val = best_v + s
             if val % 2 == 0:
                 u = val // 2
                 if 1 <= u <= N:
                     blocked_by_best.add(u)
        
        # u = 2v - s
        for s in selected[:-1]:
             u = 2 * best_v - s
             if 1 <= u <= N:
                 blocked_by_best.add(u)
        
        # u = v + s2 - s
        old_selected = selected[:-1]
        for s2 in old_selected:
            for s in old_selected:
                u = best_v + s2 - s
                if 1 <= u <= N:
                    blocked_by_best.add(u)
        
        # Also u=best_v is removed
        blocked_by_best.add(best_v)
        
        available = available - blocked_by_best
        
    return selected
# EVOLVE-BLOCK-END
