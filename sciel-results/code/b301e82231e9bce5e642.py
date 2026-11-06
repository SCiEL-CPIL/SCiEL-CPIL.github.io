import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    
    n = 512
    k = 4
    
    # Precompute all 4-APs in [1, n]
    # A 4-AP is (a, a+d, a+2d, a+3d) where a >= 1, a+3d <= n
    ap_list = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each element x, precompute which APs contain x
    # And for each AP, store the indices
    element_to_aps = {x: [] for x in range(1, n + 1)}
    for idx, ap in enumerate(ap_list):
        for elem in ap:
            element_to_aps[elem].append(idx)
    
    # Precompute bitmask for each element: which APs would be violated if this element is added
    # Actually, we need: for a given current set, which elements can be added without creating a 4-AP
    # An element x can be added if no AP containing x has all other 3 elements already in the set
    
    # Let's use a different approach: maintain a count for each AP of how many of its 4 elements are in the set
    # An element x is addable if for every AP containing x, the count of other elements in that AP is < 3
    # i.e., count[ap] <= 2 for all ap containing x (since x is not yet in the set)
    
    # Even better: maintain a "conflict" count for each element
    # conflict[x] = number of APs containing x where the other 3 elements are all in the set
    # x is addable iff conflict[x] == 0
    
    # Let's precompute for each AP, the 4 elements
    # And for each element, the list of AP indices it belongs to
    
    def try_add(x, selected_mask, ap_counts, conflict_counts):
        """Try to add element x. Returns (success, new_selected_mask, new_ap_counts, new_conflict_counts)"""
        # Check if x can be added: conflict_counts[x] must be 0
        if conflict_counts[x] > 0:
            return False, selected_mask, ap_counts, conflict_counts
        
        # x is addable. Add it.
        new_selected_mask = selected_mask.copy()
        new_selected_mask[x - 1] = True  # 0-indexed
        
        new_ap_counts = ap_counts.copy()
        new_conflict_counts = conflict_counts.copy()
        
        # For each AP containing x, increment its count
        # And update conflict counts for the other 3 elements in each such AP
        for ap_idx in element_to_aps[x]:
            new_ap_counts[ap_idx] += 1
            ap = ap_list[ap_idx]
            # The other 3 elements in this AP
            for other in ap:
                if other != x:
                    # This AP now has one more element in the set
                    # So for 'other', the number of APs where all OTHER 3 are in the set may have increased
                    # Specifically, if the other 3 elements (including x) are all in the set, then 'other' gains a conflict
                    # The other 3 elements of this AP from 'other's perspective are: x and the two others
                    # So we need to check if x and the two other non-other elements are in the set
                    # Since we're adding x, and the AP count just increased, let's check:
                    # new_ap_counts[ap_idx] is the count of elements in this AP that are in the set
                    # If new_ap_counts[ap_idx] == 4, then all 4 are in the set, which means we have a 4-AP!
                    # But wait, we're checking if x can be added. If adding x creates a 4-AP, then x shouldn't be added.
                    # So we need to check: for each AP containing x, are the other 3 already in the set?
                    # That's exactly what conflict_counts[x] > 0 means.
                    # So if conflict_counts[x] == 0, then no AP containing x has all other 3 in the set.
                    # So adding x is safe.
                    pass
        
        # Now update conflict counts for other elements
        # For each AP containing x, the other 3 elements may have their conflict counts increased
        # If the AP now has 3 elements in the set (including x), then for each of the other elements,
        # if the remaining 2 elements (besides the element itself and x) are in the set, then...
        # Actually, let's think differently.
        
        # conflict[y] = number of APs containing y where the other 3 elements are all in the set
        # When we add x, for each AP containing x, let the other 3 elements be y1, y2, y3
        # For y1: the other 3 elements in this AP are x, y2, y3
        # So y1's conflict count increases by 1 if x, y2, y3 are all in the set
        # But x is now in the set. So we need to check if y2 and y3 are in the set.
        
        for ap_idx in element_to_aps[x]:
            ap = ap_list[ap_idx]
            others = [e for e in ap if e != x]
            # Check if all others are in the set
            if all(new_selected_mask[e - 1] for e in others):
                # Then for each element in the AP, this AP contributes to their conflict count
                # But wait, we're adding x. So for x, this would be a conflict, but we already checked conflict[x] == 0
                # So this case shouldn't happen. Let's verify.
                # If all others are in the set, then before adding x, the AP had 3 elements in the set.
                # So for x, the other 3 were in the set, so conflict[x] should have been >= 1.
                # But we checked conflict[x] == 0. So this case cannot occur.
                pass
        
        # Actually, the update is more subtle. Let's just recompute conflict counts from scratch for simplicity
        # But that's O(n * avg_aps_per_element) which might be too slow.
        
        # Alternative: maintain conflict counts incrementally
        # When adding x:
        # For each AP containing x, let others = [y1, y2, y3]
        # Before adding x: ap_count = c
        # After adding x: ap_count = c + 1
        # For each yi in others:
        #   The other 3 elements in this AP from yi's perspective are: x, and the two other yj's
        #   Before: if x was not in set, then the count of {x, yj1, yj2} in set was: (0 or 1 for x) + (count of yj1, yj2 in set)
        #   After: x is in set, so count is 1 + (count of yj1, yj2 in set)
        #   yi's conflict count increases by 1 if after adding x, all of {x, yj1, yj2} are in set
        #   i.e., if yj1 and yj2 are in set (since x is now in set)
        
        for ap_idx in element_to_aps[x]:
            ap = ap_list[ap_idx]
            others = [e for e in ap if e != x]
            # For each yi in others, check if the other two (besides yi and x) are in the set
            for yi in others:
                # The other two elements in this AP besides yi and x
                other_two = [e for e in ap if e != yi and e != x]
                if all(new_selected_mask[e - 1] for e in other_two):
                    # Then yi gains a conflict from this AP
                    new_conflict_counts[yi] += 1
        
        return True, new_selected_mask, new_ap_counts, new_conflict_counts
    
    def remove_element(x, selected_mask, ap_counts, conflict_counts):
        """Remove element x. Returns (new_selected_mask, new_ap_counts, new_conflict_counts)"""
        new_selected_mask = selected_mask.copy()
        new_selected_mask[x - 1] = False
        
        new_ap_counts = ap_counts.copy()
        new_conflict_counts = conflict_counts.copy()
        
        for ap_idx in element_to_aps[x]:
            new_ap_counts[ap_idx] -= 1
            ap = ap_list[ap_idx]
            others = [e for e in ap if e != x]
            for yi in others:
                other_two = [e for e in ap if e != yi and e != x]
                if all(new_selected_mask[e - 1] for e in other_two):
                    # Before removing x, yi had a conflict from this AP (since x, and other_two were in set)
                    # After removing x, yi no longer has this conflict
                    new_conflict_counts[yi] -= 1
        
        return new_selected_mask, new_ap_counts, new_conflict_counts
    
    # Greedy initialization with most-constrained-first
    def greedy_init():
        selected_mask = np.zeros(n, dtype=bool)
        ap_counts = np.zeros(len(ap_list), dtype=np.int32)
        conflict_counts = np.zeros(n + 1, dtype=np.int32)  # 1-indexed
        
        # Start with empty set
        # At each step, add the element with the fewest conflicts (i.e., 0 conflicts) that has the smallest index? 
        # Or use a randomized approach: among elements with 0 conflicts, pick one at random
        
        remaining = list(range(1, n + 1))
        random.shuffle(remaining)
        
        while remaining:
            # Find elements with 0 conflicts
            candidates = [x for x in remaining if conflict_counts[x] == 0]
            if not candidates:
                break
            
            # Pick one at random (or the one with smallest index for determinism, but random for diversity)
            x = random.choice(candidates)
            success, selected_mask, ap_counts, conflict_counts = try_add(x, selected_mask, ap_counts, conflict_counts)
            if success:
                remaining.remove(x)
            else:
                # Shouldn't happen since we checked conflict_counts[x] == 0
                break
        
        return selected_mask, ap_counts, conflict_counts
    
    # Local search with swaps
    def local_search(selected_mask, ap_counts, conflict_counts, max_time=5.0):
        """Try to improve the set by adding elements or swapping."""
        best_mask = selected_mask.copy()
        best_ap_counts = ap_counts.copy()
        best_conflict_counts = conflict_counts.copy()
        best_size = best_mask.sum()
        
        start = time.time()
        
        while time.time() - start < max_time:
            improved = False
            
            # Try to add a random element with 0 conflicts
            unselected = [x for x in range(1, n + 1) if not best_mask[x - 1]]
            if unselected:
                random.shuffle(unselected)
                for x in unselected:
                    if conflict_counts[x] == 0:
                        success, new_mask, new_ap_counts, new_conflict_counts = try_add(x, best_mask, best_ap_counts, best_conflict_counts)
                        if success:
                            best_mask = new_mask
                            best_ap_counts = new_ap_counts
                            best_conflict_counts = new_conflict_counts
                            best_size += 1
                            improved = True
                            break
            
            if improved:
                continue
            
            # Try swaps: remove y, add x (net 0 change in size, but may enable future additions)
            # Or remove y1, y2, add x1, x2 (net 0) or remove y, add x1, x2 (net +1)
            
            # Try net +1: remove y, add x1, x2
            selected_list = [x for x in range(1, n + 1) if best_mask[x - 1]]
            unselected_list = [x for x in range(1, n + 1) if not best_mask[x - 1]]
            
            if selected_list and len(unselected_list) >= 2:
                for _ in range(20):  # Try 20 random swaps
                    if time.time() - start > max_time:
                        break
                    y = random.choice(selected_list)
                    candidates = unselected_list.copy()
                    random.shuffle(candidates)
                    x1, x2 = candidates[0], candidates[1]
                    
                    # Remove y
                    mask2, ap_counts2, conflict2 = remove_element(y, best_mask, best_ap_counts, best_conflict_counts)
                    
                    # Try to add x1
                    success1, mask3, ap_counts3, conflict3 = try_add(x1, mask2, ap_counts2, conflict2)
                    if not success1:
                        continue
                    
                    # Try to add x2
                    success2, mask4, ap_counts4, conflict4 = try_add(x2, mask3, ap_counts3, conflict3)
                    if not success2:
                        continue
                    
                    # Success! Net +1
                    best_mask = mask4
                    best_ap_counts = ap_counts4
                    best_conflict_counts = conflict4
                    best_size += 1
                    improved = True
                    break
            
            if not improved:
                # Try restart: randomize the order and do greedy again
                # But keep the best solution found so far
                break
        
        return best_mask, best_ap_counts, best_conflict_counts, best_size
    
    # Main search loop with restarts
    best_size_overall = 0
    best_mask_overall = None
    
    for restart in range(50):  # Up to 50 restarts
        if time.time() > deadline:
            break
        
        # Greedy initialization
        selected_mask, ap_counts, conflict_counts = greedy_init()
        current_size = selected_mask.sum()
        
        # Local search
        selected_mask, ap_counts, conflict_counts, current_size = local_search(
            selected_mask, ap_counts, conflict_counts, max_time=min(3.0, max(0.5, deadline - (time.time() - start_time))))
        
        if current_size > best_size_overall:
            best_size_overall = current_size
            best_mask_overall = selected_mask.copy()
        
        if time.time() > deadline - 5:
            break
    
    # Extract the set
    result = [x for x in range(1, n + 1) if best_mask_overall[x - 1]]
    
    # Verify no 4-AP (sanity check)
    result_set = set(result)
    for a in result:
        for d in range(1, (n - a) // 3 + 1):
            if (a + d) in result_set and (a + 2 * d) in result_set and (a + 3 * d) in result_set:
                # Found a 4-AP, this shouldn't happen
                # Remove the problematic element
                result.remove(a + 3 * d)
                result_set.remove(a + 3 * d)
    
    return sorted(result)
