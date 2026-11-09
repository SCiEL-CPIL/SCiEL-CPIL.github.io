import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin for 90s limit
    
    def reachable_bitmask(elements):
        """Compute bitmask of reachable sums up to 200 using bit operations."""
        mask = 1  # bit 0 set (sum 0)
        for x in elements:
            if x > 200:
                continue
            mask |= (mask << x)
            # Mask to keep only bits 0..200
            mask &= (1 << 201) - 1
        return mask
    
    def can_add(mask, x):
        """Check if adding x to a set with reachable mask would create sum 200."""
        if x > 200:
            return True
        # If sum (200-x) is reachable, then adding x creates 200
        if (mask >> (200 - x)) & 1:
            return False
        return True
    
    def add_element(mask, x):
        """Add element x to the reachable mask."""
        if x > 200:
            return mask
        new_mask = mask | (mask << x)
        return new_mask & ((1 << 201) - 1)
    
    def greedy_order(order, deadline):
        """Run greedy with given order, return selected list and mask."""
        selected = []
        mask = 1  # sum 0 is reachable
        
        for x in order:
            if time.time() > deadline:
                break
            if not can_add(mask, x):
                continue
            selected.append(x)
            mask = add_element(mask, x)
        
        return selected, mask
    
    best_selected = []
    best_size = 0
    best_mask = 1
    
    # Try descending order
    order_desc = list(range(100, 0, -1))
    sel, mask = greedy_order(order_desc, deadline)
    if len(sel) > best_size:
        best_size = len(sel)
        best_selected = sel
        best_mask = mask
    
    # Try ascending order
    if time.time() < deadline:
        order_asc = list(range(1, 101))
        sel, mask = greedy_order(order_asc, deadline)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
            best_mask = mask
    
    # Try sorted by value (many strategies)
    if time.time() < deadline:
        # Sort by: smaller first
        order_small_first = sorted(range(1, 101), key=lambda x: x)
        sel, mask = greedy_order(order_small_first, deadline)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
            best_mask = mask
    
    # Try many random orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 10.0:
        order = list(range(1, 101))
        random.shuffle(order)
        sel, mask = greedy_order(order, deadline)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
            best_mask = mask
        attempts += 1
        if attempts >= 200:
            break
    
    # Local search: try to add single elements
    if time.time() < deadline - 5.0 and best_selected:
        selected_set = set(best_selected)
        all_nums = set(range(1, 101))
        skipped = sorted(all_nums - selected_set)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try adding single skipped elements
            mask = best_mask
            for s in skipped:
                if time.time() > deadline - 2.0:
                    break
                if not can_add(mask, s):
                    continue
                # Try adding s
                best_selected.append(s)
                best_size += 1
                best_mask = add_element(best_mask, s)
                selected_set.add(s)
                skipped.remove(s)
                improved = True
                break
    
    # Local search: try removing one and adding two
    if time.time() < deadline - 3.0 and best_selected:
        selected_set = set(best_selected)
        all_nums = set(range(1, 101))
        skipped = sorted(all_nums - selected_set)
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Try removing one element and adding two skipped ones
            for rem in list(best_selected):
                if time.time() > deadline - 1.0:
                    break
                # Remove rem from best
                best_selected.remove(rem)
                selected_set.discard(rem)
                
                # Recompute mask without rem
                mask = 1
                for x in best_selected:
                    mask = add_element(mask, x)
                
                # Try to add two skipped elements
                found = False
                for i, s1 in enumerate(skipped):
                    if time.time() > deadline - 1.0:
                        break
                    if not can_add(mask, s1):
                        continue
                    new_mask1 = add_element(mask, s1)
                    for j, s2 in enumerate(skipped):
                        if j == i:
                            continue
                        if not can_add(new_mask1, s2):
                            continue
                        # Adding both is valid
                        best_selected.append(s1)
                        best_selected.append(s2)
                        best_size += 1  # removed 1, added 2, net +1
                        selected_set.add(s1)
                        selected_set.add(s2)
                        skipped.remove(s1)
                        skipped.remove(s2)
                        best_mask = add_element(new_mask1, s2)
                        improved = True
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
                # If we didn't find an improvement, put rem back
                if not found:
                    best_selected.append(rem)
                    selected_set.add(rem)
                    best_mask = best_mask  # already correct since we didn't change
    
    # Final polish: try adding more elements
    if time.time() < deadline - 1.0 and best_selected:
        selected_set = set(best_selected)
        all_nums = set(range(1, 101))
        skipped = sorted(all_nums - selected_set)
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            mask = best_mask
            for s in skipped:
                if time.time() > deadline - 0.5:
                    break
                if not can_add(mask, s):
                    continue
                best_selected.append(s)
                best_size += 1
                best_mask = add_element(best_mask, s)
                selected_set.add(s)
                skipped.remove(s)
                improved = True
                break
    
    if not best_selected:
        best_selected = [100]
    
    return best_selected
