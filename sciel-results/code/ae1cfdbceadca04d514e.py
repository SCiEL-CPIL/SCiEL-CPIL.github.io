import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _compute_ap_mask(mask, n):
    """
    For each element x in the set, compute a bitmask of 'forbidden' elements y
    such that {x, y} is part of a 3-AP already present in the set.
    Returns a list of length n+1 (1-indexed) where ap_mask[x] is an int bitmask.
    """
    ap_mask = [0] * (n + 1)
    # Iterate over all pairs (a, b) in the set with a < b
    # For each pair, the third terms are 2b-a and 2a-b
    # We can iterate a from 1 to n, b from a+1 to n
    # This is O(n^2) per computation. n=256 -> ~32k pairs.
    # We can optimize by only iterating pairs that are actually in the mask.
    
    # Let's use a simpler approach for speed: 
    # For each d in 1..n//2:
    #   For each a in 1..n-2d:
    #     b = a+d, c = a+2d
    #     If a,b,c are in set, then a conflicts with c, b conflicts with a&c, c conflicts with a.
    #     Wait, the conflict is: if a and b are in set, c is forbidden.
    #     If a and c are in set, b is forbidden.
    #     If b and c are in set, a is forbidden.
    
    # Let's just iterate all 3-APs (a, a+d, a+2d)
    # If at least two are in the set, the third is "conflicted" (forbidden if the other two stay).
    # Actually, for the local search step "remove minimal conflict", we need to know
    # which elements are part of at least one 3-AP.
    
    # Let's precompute all 3-APs as (a, b, c)
    # There are ~n^2/4 such tuples.
    # For each tuple, check membership.
    
    # To make it fast, let's use a boolean array `in_set`
    in_set = [False] * (n + 1)
    for i in range(1, n + 1):
        if mask & (1 << (i - 1)):
            in_set[i] = True
    
    # Count conflicts for each element
    conflict_count = [0] * (n + 1)
    
    max_d = n // 2
    for d in range(1, max_d + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # Check if any 2 of 3 are in set
            # If a,b in set -> c is forbidden (conflict for c? No, conflict for the set containing a,b,c)
            # We want to identify elements that are "bad".
            # An element x is "bad" if there exists a 3-AP containing x where the other two are also in the set.
            # So:
            if in_set[a] and in_set[b]:
                conflict_count[c] += 1
                conflict_count[a] += 1
                conflict_count[b] += 1
            elif in_set[a] and in_set[c]:
                conflict_count[b] += 1
                conflict_count[a] += 1
                conflict_count[c] += 1
            elif in_set[b] and in_set[c]:
                conflict_count[a] += 1
                conflict_count[b] += 1
                conflict_count[c] += 1
                
    return conflict_count

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 256
    # Initialize with the greedy solution to have a good starting point
    # Or start with a random set? Greedy is safer.
    # Let's use the greedy from the seed as the initial state.
    selected = []
    for x in range(1, n + 1):
        if all(2 * b - a != x for a in selected for b in selected if a < b):
            selected.append(x)
    
    # Convert to bitmask
    mask = 0
    for x in selected:
        mask |= (1 << (x - 1))
    
    best_score = sum(Fraction(1, x) for x in selected)
    best_mask = mask
    
    # Local Search
    # Strategy:
    # 1. While there are conflicts, remove the element with the highest conflict count (or random among max).
    #    This ensures we reach a valid set.
    # 2. Try to add elements that are not in the set. If adding an element creates conflicts, 
    #    we might need to remove others. This is complex.
    #    Simpler: 
    #    Phase 1: Remove conflicts to get a valid set.
    #    Phase 2: Try adding elements. If adding x creates a conflict, try removing one of the conflicting partners 
    #             if the net score improves.
    
    # Let's refine Phase 2:
    # For each x not in set:
    #   Calculate conflicts if x is added.
    #   If no conflicts, add x.
    #   If conflicts exist, consider removing one conflicting element y if 1/x > 1/y (i.e., x < y).
    #   This is a heuristic.
    
    # Actually, a better local search is Simulated Annealing or simple Hill Climbing on the score.
    # Move: Add a random element not in set. If invalid, remove a random element from the created 3-APs.
    # Move: Remove a random element from set.
    
    # Let's stick to a deterministic-ish hill climbing that is fast.
    # 1. Ensure validity by removing conflicts.
    # 2. Attempt to add smallest missing number. If it causes conflict, try to resolve by removing largest conflicting number.
    
    current_mask = mask
    
    # Helper to check if mask is valid and return conflict count
    def is_valid_and_count(mask_val):
        in_set = [False] * (n + 1)
        for i in range(1, n + 1):
            if mask_val & (1 << (i - 1)):
                in_set[i] = True
        
        conflict_count = [0] * (n + 1)
        max_d = n // 2
        for d in range(1, max_d + 1):
            for a in range(1, n - 2 * d + 1):
                b = a + d
                c = a + 2 * d
                if in_set[a] and in_set[b]:
                    conflict_count[c] += 1
                    conflict_count[a] += 1
                    conflict_count[b] += 1
                elif in_set[a] and in_set[c]:
                    conflict_count[b] += 1
                    conflict_count[a] += 1
                    conflict_count[c] += 1
                elif in_set[b] and in_set[c]:
                    conflict_count[a] += 1
                    conflict_count[b] += 1
                    conflict_count[c] += 1
        
        total_conflicts = sum(conflict_count)
        return total_conflicts == 0, conflict_count

    # First, ensure current_mask is valid
    valid, counts = is_valid_and_count(current_mask)
    while not valid:
        # Remove element with max conflict
        max_conf = 0
        worst_elem = -1
        for i in range(1, n + 1):
            if current_mask & (1 << (i - 1)):
                if counts[i] > max_conf:
                    max_conf = counts[i]
                    worst_elem = i
        if worst_elem == -1:
            break
        current_mask &= ~(1 << (worst_elem - 1))
        valid, counts = is_valid_and_count(current_mask)
        if time.time() > deadline:
            break
    
    # Now try to improve score
    # We will iterate through numbers 1..n.
    # If a number is not in the set, try adding it.
    # If adding it causes a conflict, see if we can remove a larger number involved in the conflict.
    
    # To speed up, we can just try to add numbers in increasing order.
    # If adding x is valid, add it.
    # If not, skip.
    # This is basically a greedy re-scan, but starting from a potentially different state.
    # To be better, we should try to "swap": if adding x creates conflict with y, and x < y, remove y and add x.
    
    # Let's implement a loop that tries to add each missing number, and if it conflicts,
    # it identifies the conflict partners and tries to swap with the largest one if beneficial.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, n + 1):
            if time.time() > deadline:
                return [i for i in range(1, n+1) if best_mask & (1 << (i-1))]
            
            if current_mask & (1 << (x - 1)):
                continue
            
            # Try adding x
            new_mask = current_mask | (1 << (x - 1))
            valid, counts = is_valid_and_count(new_mask)
            
            if valid:
                # Add it
                current_mask = new_mask
                improved = True
                # Update best if score improves
                # Score improvement is always positive if we add an element
                # But we need to track the exact score.
                # We can just recompute at the end or track incrementally.
                # Since we are only adding, score increases.
                # But we might swap later.
                pass
            else:
                # Invalid. Find conflicts.
                # The conflicts are the elements with count > 0 in the new_mask context?
                # No, `counts` from `is_valid_and_count(new_mask)` gives the conflict counts for that specific mask.
                # We need to identify which elements are "blocking" x.
                # Actually, if adding x makes it invalid, there is at least one 3-AP involving x and two existing elements.
                # Let's find the partners.
                partners = []
                for d in range(1, n // 2 + 1):
                    # x as a
                    if x + 2 * d <= n:
                        b = x + d
                        c = x + 2 * d
                        if (current_mask & (1 << (b - 1))) and (current_mask & (1 << (c - 1))):
                            partners.append(b)
                            partners.append(c)
                    # x as b
                    if x - d >= 1 and x + d <= n:
                        a = x - d
                        c = x + d
                        if (current_mask & (1 << (a - 1))) and (current_mask & (1 << (c - 1))):
                            partners.append(a)
                            partners.append(c)
                    # x as c
                    if x - 2 * d >= 1:
                        a = x - 2 * d
                        b = x - d
                        if (current_mask & (1 << (a - 1))) and (current_mask & (1 << (b - 1))):
                            partners.append(a)
                            partners.append(b)
                
                # We have a list of partners that form 3-APs with x.
                # We want to remove one partner y such that 1/x > 1/y (i.e., x < y).
                # To maximize gain, we should remove the largest possible y?
                # No, removing a large y gives a small loss (1/y is small).
                # So we should prefer to remove the largest y among the partners.
                
                if partners:
                    # Find the largest partner
                    max_partner = max(partners)
                    if max_partner > x:
                        # Swap: remove max_partner, add x
                        # Check if removing max_partner resolves the conflict?
                        # It might not resolve ALL conflicts if x has multiple 3-APs.
                        # But removing one partner removes at least one conflict.
                        # We need to ensure the resulting set is valid.
                        # This is risky. A safer swap is to only do it if it results in a valid set.
                        
                        swap_mask = (current_mask & ~(1 << (max_partner - 1))) | (1 << (x - 1))
                        valid_swap, _ = is_valid_and_count(swap_mask)
                        if valid_swap:
                            # Perform swap
                            current_mask = swap_mask
                            improved = True
                            # Score change: -1/max_partner + 1/x. Since x < max_partner, this is positive.
                            pass

    # Final validation and score calculation
    valid_final, _ = is_valid_and_count(current_mask)
    if not valid_final:
        # Fallback to best known valid
        current_mask = best_mask
        # Ensure best_mask is valid
        valid_best, _ = is_valid_and_count(best_mask)
        if not valid_best:
             # Greedy fallback
             sel = []
             for x in range(1, n+1):
                 if all(2*b-a!=x for a in sel for b in sel if a<b):
                     sel.append(x)
             return sel

    result = [i for i in range(1, n + 1) if current_mask & (1 << (i - 1))]
    
    # If result is empty, return [1]
    if not result:
        return [1]
        
    return result

# EVOLVE-BLOCK-END
