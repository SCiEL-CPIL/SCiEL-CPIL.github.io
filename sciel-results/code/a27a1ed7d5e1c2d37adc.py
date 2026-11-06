import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    # Map mask to index for quick lookup?
    # Or just use set of masks.
    
    best_family = []
    start_time = time.time()
    max_time = 55.0

    while time.time() - start_time < max_time:
        # Phase 1: Greedy Construction
        random.shuffle(all_sets)
        family = []
        family_set = set()
        pair_intersections = []
        
        for s in all_sets:
            valid = True
            for a, b, i in pair_intersections:
                if (a & s == i) and (b & s == i):
                    valid = False
                    break
            
            if valid:
                family.append(s)
                family_set.add(s)
                for a in family[:-1]:
                    i = a & s
                    pair_intersections.append((a, s, i))
        
        # Phase 2: Local Search
        # Try to add sets
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            # Find a valid set to add
            # We can iterate through all_sets not in family
            # To speed up, we can just iterate through a random subset
            
            # Let's try to add a set
            candidates = [s for s in all_sets if s not in family_set]
            random.shuffle(candidates)
            
            added = False
            for s in candidates:
                valid = True
                for a, b, i in pair_intersections:
                    if (a & s == i) and (b & s == i):
                        valid = False
                        break
                
                if valid:
                    # Add s
                    family.append(s)
                    family_set.add(s)
                    for a in family[:-1]:
                        i = a & s
                        pair_intersections.append((a, s, i))
                    improved = True
                    added = True
                    break
            
            if not added:
                # Try 2-for-1 swap
                # Pick a random set to remove
                if family:
                    out_idx = random.randint(0, len(family) - 1)
                    s_out = family[out_idx]
                    
                    # Temporarily remove s_out
                    family.pop(out_idx)
                    family_set.remove(s_out)
                    
                    # Rebuild pair_intersections? 
                    # Or just check validity against the new family.
                    # Rebuilding is O(|F|^2). |F| is small.
                    new_pair_intersections = []
                    for i in range(len(family)):
                        for j in range(i+1, len(family)):
                            a = family[i]
                            b = family[j]
                            new_pair_intersections.append((a, b, a & b))
                    
                    # Find two sets to add
                    candidates = [s for s in all_sets if s not in family_set]
                    random.shuffle(candidates)
                    
                    found_two = False
                    for s1 in candidates:
                        # Check if s1 is valid for current family
                        valid1 = True
                        for a, b, i in new_pair_intersections:
                            if (a & s1 == i) and (b & s1 == i):
                                valid1 = False
                                break
                        if not valid1:
                            continue
                        
                        # Try to find s2
                        for s2 in candidates:
                            if s2 == s1:
                                continue
                            # Check if s2 is valid for current family
                            valid2 = True
                            for a, b, i in new_pair_intersections:
                                if (a & s2 == i) and (b & s2 == i):
                                    valid2 = False
                                    break
                            if not valid2:
                                continue
                            
                            # Check if s1 and s2 are valid together with family
                            # This means no sunflower involving s1, s2 and any a in family
                            # And no sunflower involving s1, s2 and any other pair?
                            # Actually, we just need to check if {s1, s2, a} is a sunflower for any a in family.
                            # And {s1, s2, b} ...
                            # Basically, s1 and s2 must not form a sunflower with any a in family.
                            # Condition: For all a in family, (s1 & s2) != (s1 & a) != (s2 & a) ... wait.
                            # Sunflower condition: s1 & s2 = s1 & a = s2 & a.
                            
                            valid_pair = True
                            for a in family:
                                i12 = s1 & s2
                                i1a = s1 & a
                                i2a = s2 & a
                                if i12 == i1a and i1a == i2a:
                                    valid_pair = False
                                    break
                            
                            if valid_pair:
                                # Add s1 and s2
                                family.append(s1)
                                family_set.add(s1)
                                family.append(s2)
                                family_set.add(s2)
                                
                                # Rebuild pair_intersections
                                pair_intersections = []
                                for i in range(len(family)):
                                    for j in range(i+1, len(family)):
                                        a = family[i]
                                        b = family[j]
                                        pair_intersections.append((a, b, a & b))
                                
                                improved = True
                                found_two = True
                                break
                        
                        if found_two:
                            break
                    
                    if not found_two:
                        # Revert removal
                        family.insert(out_idx, s_out)
                        family_set.add(s_out)
                        # Rebuild pair_intersections
                        pair_intersections = []
                        for i in range(len(family)):
                            for j in range(i+1, len(family)):
                                a = family[i]
                                b = family[j]
                                pair_intersections.append((a, b, a & b))

        if len(family) > len(best_family):
            best_family = family
            if len(best_family) >= 250:
                break

    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
