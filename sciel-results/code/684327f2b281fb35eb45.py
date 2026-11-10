"""Construct a large 4-AP-free subset of [1, 512] using greedy + local search."""
import time

def construct():
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    
    n = 512
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with a >= 1, d >= 1
    # Store for each element, which APs it belongs to
    # For fast checking, store for each pair (i, j) with i<j, the APs they form
    
    # Better: for each element x, store the set of (d, role) such that x is in an AP
    # Actually, for local search we need to check if adding x creates a 4-AP
    # A 4-AP containing x: x can be at position 0, 1, 2, or 3
    
    # Precompute: for each x, list of (other_three_elements) that form a 4-AP with x
    # This is expensive but n=512 is small
    
    # Alternative: for each x, and each d, check the 4 positions
    # Let's precompute for each x, the set of "conflicts" - other elements that would complete an AP
    
    # For local search, we need:
    # 1. Can we add x? -> no 4-AP fully contained in S ∪ {x}
    # 2. Can we swap x (in S) with y (not in S)? -> S - {x} + {y} has no 4-AP
    
    # Precompute all 4-APs as tuples of 4 elements
    aps = []
    ap_index = {}  # element -> list of AP indices containing it
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                ap = (a, b, c, e)
                ap_index[a] = ap_index.get(a, [])
                ap_index[b] = ap_index.get(b, [])
                ap_index[c] = ap_index.get(c, [])
                ap_index[e] = ap_index.get(e, [])
                ap_index[a].append(len(aps))
                ap_index[b].append(len(aps))
                ap_index[c].append(len(aps))
                ap_index[e].append(len(aps))
                aps.append(ap)
    
    # For fast AP checking, use a boolean array
    in_set = [False] * (n + 1)
    
    def check_ap_free(selected_bool):
        """Check if selected_bool represents a 4-AP-free set."""
        for ap in aps:
            if all(selected_bool[x] for x in ap):
                return False
        return True
    
    def can_add(x, selected_bool):
        """Check if adding x to selected_bool keeps it 4-AP-free."""
        for ap_idx in ap_index.get(x, []):
            ap = aps[ap_idx]
            others = [y for y in ap if y != x]
            if all(selected_bool[y] for y in others):
                return False
        return True
    
    def can_swap_remove_add(x_out, x_in, selected_bool):
        """Check if removing x_out and adding x_in keeps it 4-AP-free."""
        # After removal, x_out is not in set. We need to check all APs.
        # An AP is violated if all 4 are in the new set.
        # New set = old set - {x_out} + {x_in}
        # An AP is in the new set iff:
        #   - All 4 elements are in old set, and none is x_out (so x_out not in AP), 
        #     OR x_in is in AP and the other 3 are in old set
        # More carefully:
        # For an AP to be fully in new set:
        #   Case 1: x_in in AP, x_out not in AP, other 3 of AP in old set
        #   Case 2: x_in not in AP, x_out not in AP, all 4 in old set (already checked, but x_out not in AP so fine)
        #   Case 3: x_in in AP, x_out in AP -> need other 2 in old set, and x_out was the 4th... 
        # Actually let's just check directly for APs that could be newly formed.
        
        # An AP is violated in new set iff all 4 elements are in new set.
        # new_set = old_set - {x_out} + {x_in}
        # For AP to be in new_set:
        #   - If x_out in AP: need x_in in AP AND other 2 (besides x_out, x_in) in old_set
        #     (if x_in == x_out, no change, but we assume x_in != x_out)
        #   - If x_out not in AP: need x_in in AP AND other 3 in old_set, OR x_in not in AP AND all 4 in old_set
        #     But if x_in not in AP and all 4 in old_set, that AP was already violated (shouldn't happen)
        #     So: x_in in AP, x_out not in AP, other 3 in old_set
        
        # So we only need to check APs containing x_in:
        for ap_idx in ap_index.get(x_in, []):
            ap = aps[ap_idx]
            if x_out in ap:
                # Need other 2 elements (besides x_out and x_in) to be in old set
                others = [y for y in ap if y != x_out and y != x_in]
                if all(selected_bool[y] for y in others):
                    return False
            else:
                # x_out not in AP, need all other 3 (besides x_in) in old set
                others = [y for y in ap if y != x_in]
                if all(selected_bool[y] for y in others):
                    return False
        return True
    
    # Phase 1: Greedy construction
    selected = set()
    for x in range(1, n + 1):
        if can_add(x, in_set):
            selected.add(x)
            in_set[x] = True
    
    best_size = len(selected)
    best_set = set(selected)
    
    # Phase 2: Local search with 1- and 2-exchanges
    # Try to add any element not in set
    # Try to swap one element out for one in
    
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        
        # Try 1-add: add any element not in selected
        for x in range(1, n + 1):
            if (time.time() - start_time) > DEADLINE:
                break
            if x not in selected:
                if can_add(x, in_set):
                    selected.add(x)
                    in_set[x] = True
                    improved = True
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_set = set(selected)
        
        if (time.time() - start_time) > DEADLINE:
            break
        
        # Try 1-swap: remove x_out, add x_in
        selected_list = sorted(selected)
        for x_out in selected_list:
            if (time.time() - start_time) > DEADLINE:
                break
            for x_in in range(1, n + 1):
                if x_in == x_out or x_in in selected:
                    continue
                if can_swap_remove_add(x_out, x_in, in_set):
                    # Do the swap
                    selected.remove(x_out)
                    in_set[x_out] = False
                    selected.add(x_in)
                    in_set[x_in] = True
                    improved = True
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_set = set(selected)
                    break  # restart outer loop after swap
    
    # Phase 3: More aggressive 2-exchange local search
    # Try swapping two elements out for two elements in
    for iteration in range(200):
        if (time.time() - start_time) > DEADLINE:
            break
        found_swap = False
        selected_list = sorted(selected)
        for i in range(len(selected_list)):
            if (time.time() - start_time) > DEADLINE:
                break
            x_out1 = selected_list[i]
            for j in range(i + 1, len(selected_list)):
                x_out2 = selected_list[j]
                for x_in1 in range(1, n + 1):
                    if (time.time() - start_time) > DEADLINE:
                        break
                    if x_in1 in selected or x_in1 == x_out1 or x_in1 == x_out2:
                        continue
                    for x_in2 in range(x_in1 + 1, n + 1):
                        if (time.time() - start_time) > DEADLINE:
                            break
                        if x_in2 in selected or x_in2 == x_out1 or x_in2 == x_out2:
                            continue
                        # Check if removing x_out1, x_out2 and adding x_in1, x_in2 is valid
                        # New set = old - {x_out1, x_out2} + {x_in1, x_in2}
                        # Check all APs: an AP is violated if all 4 in new set
                        # Only need to check APs containing x_in1 or x_in2
                        valid = True
                        checked_aps = set()
                        for x_in in (x_in1, x_in2):
                            for ap_idx in ap_index.get(x_in, []):
                                if ap_idx in checked_aps:
                                    continue
                                checked_aps.add(ap_idx)
                                ap = aps[ap_idx]
                                # Count how many of the 4 are in new set
                                # x_out1, x_out2 are removed; x_in1, x_in2 are added
                                count = 0
                                for y in ap:
                                    if y == x_out1 or y == x_out2:
                                        continue  # not in new set
                                    if y == x_in1 or y == x_in2:
                                        count += 1
                                    elif in_set[y]:
                                        count += 1
                                if count == 4:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Do the swap
                            selected.remove(x_out1)
                            in_set[x_out1] = False
                            selected.remove(x_out2)
                            in_set[x_out2] = False
                            selected.add(x_in1)
                            in_set[x_in1] = True
                            selected.add(x_in2)
                            in_set[x_in2] = True
                            found_swap = True
                            if len(selected) > best_size:
                                best_size = len(selected)
                                best_set = set(selected)
                            break
                    if found_swap:
                        break
                if found_swap:
                    break
            if found_swap:
                break
        if not found_swap:
            break
    
    # Return the best set found
    result = sorted(best_set)
    if not result:
        result = [1]
    return result
