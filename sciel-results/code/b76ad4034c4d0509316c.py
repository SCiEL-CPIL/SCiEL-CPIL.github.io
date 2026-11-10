"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    # Distinct approach: Greedy construction maximizing the number of 'blocked' elements
    # at each step to achieve maximality with minimal cardinality.
    
    def check_sidon_addition(x, selected, sums):
        """Check if x can be added to selected without violating Sidon property."""
        new_sums = set()
        for a in selected:
            s = x + a
            if s in sums or s in new_sums:
                return False
            new_sums.add(s)
        s_2x = 2 * x
        if s_2x in sums or s_2x in new_sums:
            return False
        new_sums.add(s_2x)
        return True

    def get_blocked_elements(selected, sums):
        """Return set of elements in [1, N] not in selected that cannot be added."""
        blocked = set()
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if x can be added
            can_add = True
            for a in selected:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add and (2 * x) in sums:
                can_add = False
            if not can_add:
                blocked.add(x)
        return blocked

    # Greedy construction
    selected = []
    sums = set()
    blocked = set()
    
    # We iterate until all elements are either in selected or blocked
    # But we must ensure we don't get stuck with a non-maximal set.
    # The greedy choice should ensure maximality if we run until no candidates remain.
    
    remaining_candidates = set(range(1, N + 1))
    
    while remaining_candidates:
        best_x = None
        max_new_blocks = -1
        
        for x in remaining_candidates:
            if not check_sidon_addition(x, selected, sums):
                continue
            
            # Simulate adding x
            # We need to calculate how many *currently unblocked* elements would become blocked
            # Or simply, how many total elements would be blocked after adding x.
            
            # Efficiently calculating new blocked set is expensive if done naively for each x.
            # Optimization: 
            # Adding x creates new sums: {x+a for a in selected} U {2x}.
            # An element y is blocked if it conflicts with any sum in S_new.
            # Conflict for y: exists a in selected U {x} such that y+a in S_new, or 2y in S_new.
            
            # Let's just compute the count of newly blocked elements among those currently in remaining_candidates - {x}
            # Actually, simpler: just count how many elements from [1, N] \ (selected U {x}) are blocked by (selected U {x}).
            
            # To speed up, we can approximate or just do the full check for the top few candidates.
            # Given N=256, and small selected size, full check is O(N * |selected|).
            # |selected| will be small (< 20). N=256. 256*20 = 5k ops. 
            # Loop over candidates (up to 256). 256 * 5k = 1.2M ops. Fast enough.
            
            new_selected = selected + [x]
            new_sums = set(sums)
            for a in selected:
                new_sums.add(x + a)
            new_sums.add(2 * x)
            
            count_blocked = 0
            for y in range(1, N + 1):
                if y in new_selected:
                    continue
                # Check if y is blocked by new_selected/new_sums
                is_blocked = False
                for a in new_selected:
                    if (y + a) in new_sums:
                        is_blocked = True
                        break
                if not is_blocked and (2 * y) in new_sums:
                    is_blocked = True
                
                if is_blocked:
                    count_blocked += 1
            
            if count_blocked > max_new_blocks:
                max_new_blocks = count_blocked
                best_x = x
        
        if best_x is None:
            # No candidate can be added? This shouldn't happen if we started empty and N is large enough
            # to allow a Sidon set. But if we are stuck, we break.
            break
        
        # Commit best_x
        selected.append(best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
        sums.add(2 * best_x)
        
        remaining_candidates.remove(best_x)
        
        # Update blocked set for next iteration's potential pruning?
        # We don't strictly need to maintain 'blocked' if we just check 'remaining_candidates' 
        # and filter by Sidon check. But we can prune remaining_candidates to those 
        # that are still Sidon-compatible? No, a candidate might become compatible later? 
        # No, adding elements only creates more sums, so if x is not Sidon-compatible now, 
        # it will never be Sidon-compatible later.
        # So we can prune remaining_candidates to only those that pass check_sidon_addition.
        
        new_remaining = set()
        for x in remaining_candidates:
            if check_sidon_addition(x, selected, sums):
                new_remaining.add(x)
        remaining_candidates = new_remaining
        
        # Early stop if all remaining candidates are blocked?
        # The loop continues as long as there are candidates that *can* be added (Sidon-wise).
        # If we stop, the set is maximal? 
        # If no x in remaining_candidates can be added due to Sidon constraint, 
        # does that mean the set is maximal?
        # Maximal means NO x in [1, N] \ selected can be added.
        # The candidates in remaining_candidates are those that *can* be added Sidon-wise.
        # If we stop when remaining_candidates is empty, it means no element can be added Sidon-wise.
        # This implies the set is maximal (since any addable element must be Sidon-compatible).
        
    # Verify maximality just in case
    if not _is_maximal(selected, sums):
        # Fallback to simple greedy if heuristic failed (should not happen)
        selected, sums = [], set()
        for x in range(1, N + 1):
            if check_sidon_addition(x, selected, sums):
                selected.append(x)
                for a in selected[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
        
    return selected

def _is_maximal(selected, sums):
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        can_add = True
        for a in selected:
            if (x + a) in sums:
                can_add = False
                break
        if can_add and (2 * x) in sums:
            can_add = False
        if can_add:
            return False
    return True
# EVOLVE-BLOCK-END
