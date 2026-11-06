import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    # Generate all k-subsets as bitmasks
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    best_family = []
    start_time = time.time()
    max_time = 50.0 # Leave buffer for overhead
    
    def can_add(s, pair_intersections):
        for a, b, i in pair_intersections:
            if (a & s == i) and (b & s == i):
                return False
        return True

    def add_set(s, family, pair_intersections):
        family.append(s)
        for a in family[:-1]:
            i = a & s
            pair_intersections.append((a, s, i))

    while time.time() - start_time < max_time:
        random.shuffle(all_sets)
        family = []
        pair_intersections = []
        
        # Greedy Phase
        for s in all_sets:
            if can_add(s, pair_intersections):
                add_set(s, family, pair_intersections)
        
        # Local Search Phase: Try to add more sets
        # We use a set for fast membership check
        family_set = set(family)
        
        # Attempt to add any remaining set
        # Shuffle again to try different orders for additions
        random.shuffle(all_sets)
        for s in all_sets:
            if s in family_set:
                continue
            if can_add(s, pair_intersections):
                add_set(s, family, pair_intersections)
                family_set.add(s)
        
        if len(family) > len(best_family):
            best_family = family.copy()
        
        # Optional: Simple Swap Local Search
        # If time permits, try swaps to potentially unlock more additions
        if time.time() - start_time < max_time - 5.0:
            for _ in range(50): # Limit swaps to avoid timeout
                if time.time() - start_time >= max_time - 5.0:
                    break
                if not family:
                    break
                
                s_out = random.choice(family)
                # Find a candidate s_in not in family
                found_swap = False
                for _ in range(10):
                    s_in = random.choice(all_sets)
                    if s_in == s_out:
                        continue
                    
                    # Temporarily remove s_out
                    # Rebuild pair_intersections without s_out
                    # This is expensive, so we do it carefully
                    new_pairs = []
                    for a, b, i in pair_intersections:
                        if a != s_out and b != s_out:
                            new_pairs.append((a, b, i))
                    
                    old_pairs = pair_intersections
                    old_family = family
                    pair_intersections = new_pairs
                    family = [x for x in family if x != s_out]
                    
                    if can_add(s_in, pair_intersections):
                        add_set(s_in, family, pair_intersections)
                        found_swap = True
                        break
                    else:
                        # Revert
                        pair_intersections = old_pairs
                        family = old_family
                        break
                
                if found_swap:
                    if len(family) > len(best_family):
                        best_family = family.copy()
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
