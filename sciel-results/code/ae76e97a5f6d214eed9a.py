import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Candidates are b = 512 - a, so a = 512 - b.
    # We want to maximize sum(1/b) for b in [1, 511]
    # subject to gcd(512-b_i, 512-b_j) == 1 for all i != j.
    
    all_bs = list(range(1, 512))
    
    # Precompute a for each b
    a_map = {b: 512 - b for b in all_bs}
    
    # Helper to check if a candidate b can be added to a set of b's
    def can_add(b, current_bs):
        a_new = a_map[b]
        for b_old in current_bs:
            a_old = a_map[b_old]
            if math.gcd(a_new, a_old) != 1:
                return False
        return True
    
    # Helper to calculate score
    def calc_score(bs):
        return sum(1.0 / b for b in bs)
        
    # Greedy Initialization
    selected = []
    for b in sorted(all_bs, key=lambda x: 1.0/x, reverse=True): # Sort by value 1/b desc, i.e., b asc
        if can_add(b, selected):
            selected.append(b)
            
    best_score = calc_score(selected)
    best_set = set(selected)
    
    # Local Search / Hill Climbing
    # We will try swaps: remove one from selected, add one from unselected
    # Also try adding one if possible
    # And removing one if it allows adding two better ones? (Harder)
    
    # For simplicity and speed, single swap and single add/remove
    unselected = [b for b in all_bs if b not in best_set]
    
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        
        # Try adding any unselected item
        for b in unselected:
            if can_add(b, best_set):
                # If we can add it, it must increase score (1/b > 0)
                new_set = best_set | {b}
                new_score = calc_score(new_set)
                if new_score > best_score:
                    best_score = new_score
                    best_set = new_set
                    unselected.remove(b)
                    improved = True
                    break # Restart search with new state
        
        # Try single swap: remove s from best_set, add u from unselected
        # Only if gcd allows u to be added to (best_set - {s})
        # And ideally 1/u > 1/s (u < s) to increase score, but even if u > s, 
        # maybe it unblocks other additions? No, we are just swapping.
        # If we swap s with u, score change is 1/u - 1/s.
        # So we only want to swap if u < s (1/u > 1/s).
        # But wait, if u is compatible with the rest, and s is blocking some other high value v?
        # That would be a double move.
        # For now, let's just do simple swaps that directly increase score.
        
        for s in list(best_set):
            for u in unselected:
                if u >= s: # 1/u <= 1/s, no direct gain
                    continue
                # Check if u is compatible with best_set - {s}
                temp_set = best_set - {s}
                if can_add(u, temp_set):
                    new_set = temp_set | {u}
                    new_score = calc_score(new_set)
                    if new_score > best_score:
                        best_score = new_score
                        best_set = new_set
                        if s in unselected:
                            pass # s is already in unselected? No, s was in best_set.
                        if s not in unselected:
                            unselected.append(s)
                        if u in unselected:
                            unselected.remove(u)
                        improved = True
                        break
            if improved:
                break
        
        # Try removing an item? 
        # Removing an item decreases score, so only useful if it allows adding many.
        # We can try: remove s, then run greedy add on the remaining unselected.
        # This is expensive. Let's skip for now or do it sparingly.
        
        # Let's try a "drop and refill" strategy for the worst item
        # Find the item s in best_set with the smallest 1/s (largest s)
        # Remove it, then try to add as many unselected items as possible greedily
        if (time.time() < deadline - 10):
            # Sort best_set by 1/s (ascending, i.e., s descending)
            # Actually, let's just try dropping the largest s
            s_to_drop = max(best_set)
            temp_set = best_set - {s_to_drop}
            temp_score = calc_score(temp_set)
            
            # Greedy add from unselected
            # We need to be careful not to add items that were already rejected?
            # Just try to add any compatible item from unselected
            added_any = True
            while added_any and (time.time() < deadline - 10):
                added_any = False
                for u in unselected:
                    if can_add(u, temp_set):
                        temp_set.add(u)
                        temp_score += 1.0 / u
                        unselected.remove(u)
                        added_any = True
                        break
            
            if temp_score > best_score:
                best_score = temp_score
                best_set = temp_set
                # Rebuild unselected
                unselected = [b for b in all_bs if b not in best_set]
                improved = True
                break

    # Convert back to a's
    result = [a_map[b] for b in best_set]
    
    return result
