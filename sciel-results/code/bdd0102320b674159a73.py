"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _make_bitmasks():
    """Generate all 4-element subsets of [1,16] as bitmasks, indexed."""
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    return all_sets

def _is_sunflower(a, b, c):
    """Check if three 4-bitmasks form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _check_conflict_fast(mask, chosen, chosen_set):
    """Check if adding mask to chosen creates a sunflower triple using bit operations."""
    for i in range(len(chosen)):
        a = chosen[i]
        ab = a & mask
        for j in range(i + 1, len(chosen)):
            b = chosen[j]
            ac = a & b
            bc = b & mask
            if ab == ac == bc:
                return True
    return False

def _greedy_build(all_masks, order, deadline):
    """Greedy: add masks in given order if no conflict."""
    chosen = []
    chosen_set = set()
    for mask in order:
        if time.time() > deadline:
            break
        # Check conflict
        conflict = False
        for i in range(len(chosen)):
            a = chosen[i]
            ab = a & mask
            for j in range(i + 1, len(chosen)):
                b = chosen[j]
                ac = a & b
                bc = b & mask
                if ab == ac == bc:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            chosen.append(mask)
            chosen_set.add(mask)
    return chosen

def _local_search(chosen, all_masks, deadline):
    """Try to improve by adding sets or swapping."""
    chosen_set = set(chosen)
    chosen = list(chosen)
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try adding random unchosen masks
        unchosen = [m for m in all_masks if m not in chosen_set]
        if not unchosen:
            break
        random.shuffle(unchosen)
        
        added_this_round = 0
        for mask in unchosen:
            if time.time() > deadline - 2:
                break
            # Check conflict
            conflict = False
            for i in range(len(chosen)):
                a = chosen[i]
                ab = a & mask
                for j in range(i + 1, len(chosen)):
                    b = chosen[j]
                    ac = a & b
                    bc = b & mask
                    if ab == ac == bc:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                chosen.append(mask)
                chosen_set.add(mask)
                improved = True
                added_this_round += 1
                if added_this_round >= 5:
                    break
        
        if not improved:
            # Try swaps: remove one, try to add two
            if len(chosen) < 100:
                break
            found_swap = False
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                i = random.randint(0, len(chosen) - 1)
                removed = chosen[i]
                chosen_temp = chosen[:i] + chosen[i+1:]
                chosen_temp_set = set(chosen_temp)
                
                unchosen = [m for m in all_masks if m not in chosen_temp_set]
                random.shuffle(unchosen)
                
                added = 0
                new_chosen = list(chosen_temp)
                new_set = set(chosen_temp)
                
                for mask in unchosen[:30]:
                    conflict = False
                    for k in range(len(new_chosen)):
                        a = new_chosen[k]
                        ab = a & mask
                        for l in range(k + 1, len(new_chosen)):
                            b = new_chosen[l]
                            ac = a & b
                            bc = b & mask
                            if ab == ac == bc:
                                conflict = True
                                break
                        if conflict:
                            break
                    if not conflict:
                        new_chosen.append(mask)
                        new_set.add(mask)
                        added += 1
                        if added >= 2:
                            break
                
                if added >= 2:
                    chosen = new_chosen
                    chosen_set = new_set
                    found_swap = True
                    improved = True
                    break
            
            if not improved:
                break
    
    return chosen

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    all_masks = _make_bitmasks()
    random.seed(42)
    
    best = []
    best_len = 0
    
    # Multiple greedy passes with different orderings
    for trial in range(20):
        if time.time() > deadline - 15:
            break
        
        order = all_masks[:]
        random.shuffle(order)
        
        chosen = _greedy_build(all_masks, order, deadline - 15)
        
        if len(chosen) > best_len:
            best = chosen[:]
            best_len = len(chosen)
        
        # Local search improvement on best
        if time.time() > deadline - 10:
            break
        improved = _local_search(best, all_masks, deadline - 5)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    # Final local search on best
    if time.time() < deadline - 3:
        improved = _local_search(best, all_masks, deadline - 1)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best:
        elems = [i+1 for i in range(16) if mask & (1 << i)]
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
