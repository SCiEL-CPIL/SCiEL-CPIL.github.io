"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    N = 256
    
    # Precompute all 3-APs and 4-APs for efficiency
    # 3-AP: (a, a+d, a+2d) with d >= 1
    three_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                three_aps.append((a, b, c))
    
    # 4-AP: (a, a+d, a+2d, a+3d) with d >= 1
    four_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                four_aps.append((a, b, c, e))
    
    # For each element, precompute which 3-APs it participates in
    elem_to_3aps = {x: [] for x in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(three_aps):
        elem_to_3aps[a].append(idx)
        elem_to_3aps[b].append(idx)
        elem_to_3aps[c].append(idx)
    
    # For each element, precompute which 4-APs it participates in
    elem_to_4aps = {x: [] for x in range(1, N + 1)}
    for idx, (a, b, c, e) in enumerate(four_aps):
        elem_to_4aps[a].append(idx)
        elem_to_4aps[b].append(idx)
        elem_to_4aps[c].append(idx)
        elem_to_4aps[e].append(idx)
    
    def count_3aps(selected_set):
        """Count number of 3-APs in selected_set."""
        count = 0
        for (a, b, c) in three_aps:
            if a in selected_set and b in selected_set and c in selected_set:
                count += 1
        return count
    
    def has_4ap(selected_set):
        """Check if selected_set contains any 4-AP."""
        for (a, b, c, e) in four_aps:
            if a in selected_set and b in selected_set and c in selected_set and e in selected_set:
                return True
        return False
    
    def greedy_construct(order):
        """Greedy construction: add elements in given order if no 4-AP is formed."""
        selected = set()
        for x in order:
            # Check if adding x creates a 4-AP
            creates_4ap = False
            for ap_idx in elem_to_4aps[x]:
                a, b, c, e = four_aps[ap_idx]
                others = [y for y in (a, b, c, e) if y != x]
                if all(y in selected for y in others):
                    creates_4ap = True
                    break
            if not creates_4ap:
                selected.add(x)
        return selected
    
    def local_search(selected, time_limit):
        """Improve 3-AP count via single-element swaps while maintaining no 4-AP."""
        selected = set(selected)
        current_count = count_3aps(selected)
        best_count = current_count
        best_set = set(selected)
        
        # Build adjacency: for each element, which other elements would conflict via 4-AP
        # If we remove x and add y, we need to check no 4-AP is formed
        
        def delta_3aps_remove_add(selected, rem, add):
            """Compute change in 3-AP count when removing rem and adding add."""
            # 3-APs lost: those containing rem that also contain at least one other selected element
            # 3-APs gained: those containing add where all other elements are in (selected - {rem})
            
            lost = 0
            for ap_idx in elem_to_3aps[rem]:
                a, b, c = three_aps[ap_idx]
                others = [y for y in (a, b, c) if y != rem]
                if all(y in selected for y in others):
                    lost += 1
            
            gained = 0
            new_selected = selected - {rem}
            for ap_idx in elem_to_3aps[add]:
                a, b, c = three_aps[ap_idx]
                others = [y for y in (a, b, c) if y != add]
                if all(y in new_selected for y in others):
                    gained += 1
            
            return gained - lost
        
        def would_create_4ap(selected, rem, add):
            """Check if replacing rem with add creates a 4-AP."""
            new_selected = selected - {rem}
            # Check 4-APs containing add
            for ap_idx in elem_to_4aps[add]:
                a, b, c, e = four_aps[ap_idx]
                others = [y for y in (a, b, c, e) if y != add]
                if all(y in new_selected for y in others):
                    return True
            return False
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            improved = False
            elements = list(selected)
            candidates = [x for x in range(1, N + 1) if x not in selected]
            
            random.shuffle(elements)
            random.shuffle(candidates)
            
            for rem in elements:
                if time.time() >= time_limit:
                    break
                for add in candidates:
                    if time.time() >= time_limit:
                        break
                    if would_create_4ap(selected, rem, add):
                        continue
                    delta = delta_3aps_remove_add(selected, rem, add)
                    if delta > 0:
                        selected.discard(rem)
                        selected.add(add)
                        current_count += delta
                        if current_count > best_count:
                            best_count = current_count
                            best_set = set(selected)
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
        
        return best_set, best_count
    
    def two_swap_search(selected, time_limit):
        """Try 2-swap moves: remove two elements, add two different elements."""
        selected = set(selected)
        current_count = count_3aps(selected)
        best_count = current_count
        best_set = set(selected)
        
        def would_create_4ap_2swap(selected, rems, adds):
            """Check if removing rems and adding adds creates any 4-AP."""
            new_selected = (selected - set(rems)) | set(adds)
            for (a, b, c, e) in four_aps:
                if a in new_selected and b in new_selected and c in new_selected and e in new_selected:
                    return True
            return False
        
        def count_3aps_fast(selected_set):
            count = 0
            for (a, b, c) in three_aps:
                if a in selected_set and b in selected_set and c in selected_set:
                    count += 1
            return count
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < time_limit and no_improve < max_no_improve:
            improved = False
            elements = list(selected)
            candidates = [x for x in range(1, N + 1) if x not in selected]
            
            random.shuffle(elements)
            random.shuffle(candidates)
            
            # Try limited number of 2-swap combinations
            tried = 0
            max_tried = 200
            
            for i in range(len(elements)):
                if time.time() >= time_limit or tried >= max_tried:
                    break
                for j in range(i + 1, len(elements)):
                    if time.time() >= time_limit or tried >= max_tried:
                        break
                    rem1, rem2 = elements[i], elements[j]
                    for k in range(len(candidates)):
                        if time.time() >= time_limit or tried >= max_tried:
                            break
                        for l in range(k + 1, len(candidates)):
                            if time.time() >= time_limit or tried >= max_tried:
                                break
                            add1, add2 = candidates[k], candidates[l]
                            tried += 1
                            
                            rems = (rem1, rem2)
                            adds = (add1, add2)
                            
                            if would_create_4ap_2swap(selected, rems, adds):
                                continue
                            
                            new_selected = (selected - set(rems)) | set(adds)
                            new_count = count_3aps_fast(new_selected)
                            
                            if new_count > current_count:
                                selected = new_selected
                                current_count = new_count
                                if current_count > best_count:
                                    best_count = current_count
                                    best_set = set(selected)
                                improved = True
                                no_improve = 0
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        return best_set, best_count
    
    # Multiple restarts with different orderings
    best_overall = set()
    best_overall_count = 0
    
    # Residue class orderings to explore different structures
    orderings = []
    
    # Natural order
    orderings.append(list(range(1, N + 1)))
    
    # Reverse order
    orderings.append(list(range(N, 0, -1)))
    
    # Residue mod 3 orderings
    for r in range(3):
        orderings.append([x for x in range(1, N + 1) if x % 3 == r] + 
                         [x for x in range(1, N + 1) if x % 3 != r])
    
    # Residue mod 4 orderings
    for r in range(4):
        orderings.append([x for x in range(1, N + 1) if x % 4 == r] + 
                         [x for x in range(1, N + 1) if x % 4 != r])
    
    # Residue mod 5 orderings
    for r in range(5):
        orderings.append([x for x in range(1, N + 1) if x % 5 == r] + 
                         [x for x in range(1, N + 1) if x % 5 != r])
    
    # Random orderings
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        orderings.append(order)
    
    # Even/odd splits
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    orderings.append(evens + odds)
    orderings.append(odds + evens)
    
    random.shuffle(orderings)
    
    for ordering in orderings:
        if time.time() >= deadline - 10:
            break
        
        # Greedy construction
        selected = greedy_construct(ordering)
        
        if not selected:
            continue
        
        # Verify no 4-AP
        if has_4ap(selected):
            # Repair: remove elements involved in 4-APs
            changed = True
            while changed:
                changed = False
                for (a, b, c, e) in four_aps:
                    if a in selected and b in selected and c in selected and e in selected:
                        # Remove the element that contributes least to 3-APs
                        # For simplicity, remove 'a'
                        selected.discard(a)
                        changed = True
                        break
        
        if not selected:
            continue
        
        # Local search with single swaps
        time_for_ls = min(time.time() + 15, deadline - 5)
        selected, count = local_search(selected, time_for_ls)
        
        if count > best_overall_count:
            best_overall_count = count
            best_overall = set(selected)
        
        # Two-swap search if time permits
        if time.time() < deadline - 15:
            time_for_2swap = min(time.time() + 10, deadline - 5)
            selected2, count2 = two_swap_search(selected, time_for_2swap)
            if count2 > best_overall_count:
                best_overall_count = count2
                best_overall = set(selected2)
        
        if time.time() >= deadline - 5:
            break
    
    # Final verification and return
    if not best_overall:
        # Fallback: simple greedy
        best_overall = greedy_construct(list(range(1, N + 1)))
        if not best_overall:
            best_overall = {1}
    
    # Ensure no 4-APs (final safety check)
    for (a, b, c, e) in four_aps:
        if a in best_overall and b in best_overall and c in best_overall and e in best_overall:
            best_overall.discard(a)
    
    if not best_overall:
        best_overall = {1}
    
    return sorted(best_overall)
# EVOLVE-BLOCK-END
