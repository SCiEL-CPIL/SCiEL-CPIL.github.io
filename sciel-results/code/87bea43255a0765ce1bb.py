"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    def get_blocked(selected, sums):
        """
        Returns a set of integers in [1, N] that cannot be added to 'selected'
        without violating the Sidon property.
        """
        blocked = set()
        selected_set = set(selected)
        
        # An element x is blocked if:
        # 1. 2*x is in sums
        # 2. x + a is in sums for any a in selected
        # Note: We only care about x not in selected.
        
        # Optimization: 
        # If x is such that 2x is in sums, x is blocked.
        # If x + a is in sums for some a, x is blocked.
        
        # Let's iterate through potential x in [1, N]
        # To be efficient, we can derive blocked elements from sums.
        # If s is in sums, then:
        # - If s is even, s/2 is blocked (if s/2 not in selected)
        # - If s - a is in [1, N] and s - a not in selected, then s - a is blocked.
        
        for s in sums:
            # Case 1: Diagonal collision 2*x = s
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= N and x not in selected_set:
                    blocked.add(x)
            
            # Case 2: Off-diagonal collision x + a = s => x = s - a
            for a in selected:
                x = s - a
                if 1 <= x <= N and x not in selected_set:
                    blocked.add(x)
        
        return blocked

    def is_sidon_and_maximal(selected):
        sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                s = selected[i] + selected[j]
                if s in sums:
                    return False, False
                sums.add(s)
        
        blocked = get_blocked(selected, sums)
        selected_set = set(selected)
        
        # Check if every element in [1, N] is either in selected or blocked
        for x in range(1, N + 1):
            if x not in selected_set and x not in blocked:
                return True, False # Sidon but not maximal
        
        return True, True

    # Deterministic Greedy with Impact Maximization
    selected = []
    sums = set()
    
    # We will iteratively pick the element that blocks the most "free" elements.
    # Free elements are those in [1, N] not in selected and not currently blocked.
    
    while True:
        is_sidon, is_max = is_sidon_and_maximal(selected)
        if not is_sidon:
            # Should not happen if we pick carefully, but safety check
            break
        if is_max:
            break
        
        # Calculate current blocked set
        current_blocked = get_blocked(selected, sums)
        selected_set = set(selected)
        
        # Find candidates: elements in [1, N] not in selected and not blocked
        # Wait, if an element is not blocked, it *can* be added.
        # We want to add one. Which one?
        # We want to minimize the final size.
        # Adding an element x adds new sums, which block more elements.
        # So we want to pick x that maximizes the number of *newly* blocked elements.
        
        best_x = -1
        max_new_blocks = -1
        
        # Optimization: Only check a subset of candidates if N is large, but N=256 is small.
        # However, checking all 256 every step might be slow if we do it many times.
        # Let's iterate through candidates.
        
        candidates = [x for x in range(1, N + 1) if x not in selected_set and x not in current_blocked]
        
        if not candidates:
            # This implies it's maximal? No, if there are no candidates, it means all are blocked or selected.
            # So it is maximal.
            break
        
        for x in candidates:
            # Simulate adding x
            new_sums = set(sums)
            # Check if x can be added (it should be, as it's a candidate)
            # New sums: 2*x and x+a for a in selected
            valid = True
            added_sums = [2*x]
            for a in selected:
                added_sums.append(x + a)
            
            # Check for collisions with existing sums
            for s in added_sums:
                if s in new_sums:
                    valid = False
                    break
            if not valid:
                # This shouldn't happen for a candidate, but if it does, skip
                continue
                
            new_sums.update(added_sums)
            
            # Calculate new blocked set
            new_blocked = get_blocked(selected + [x], new_sums)
            
            # Count how many *new* elements are blocked
            # New blocks are in new_blocked but not in current_blocked
            # Note: x itself is now in selected, so it's not "blocked" in the sense of "cannot be added", 
            # but it's removed from the pool of candidates.
            # We want to maximize the reduction in the set of "free" elements.
            # Free elements before: [1..N] \ (selected U current_blocked)
            # Free elements after: [1..N] \ ((selected U {x}) U new_blocked)
            
            # Let's just count the size of new_blocked \ current_blocked
            # This is a good heuristic.
            new_blocks = len(new_blocked - current_blocked)
            
            if new_blocks > max_new_blocks:
                max_new_blocks = new_blocks
                best_x = x
        
        if best_x == -1:
            # Fallback: pick the smallest candidate
            best_x = candidates[0]
        
        # Add best_x
        selected.append(best_x)
        # Update sums
        added_sums = [2*best_x]
        for a in selected[:-1]:
            added_sums.append(best_x + a)
        sums.update(added_sums)
        
        # Safety check for Sidon property
        # The logic above ensures no collision with *previous* sums.
        # But we must ensure no collision *within* added_sums?
        # added_sums are 2x and x+a. 
        # 2x == x+a => x=a (impossible as x not in selected)
        # x+a == x+b => a=b (impossible)
        # x+a == x+b is not possible.
        # So added_sums are distinct.
        # And we checked against existing sums.
        # So Sidon property holds.

    return selected
# EVOLVE-BLOCK-END
