"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s safety margin before 60s limit
    
    # Generate all 4-element subsets of [1,16] as bitmasks (0-indexed bits for 0..15)
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    
    def greedy_random(seed=42):
        random.seed(seed)
        indices = list(range(n_total))
        random.shuffle(indices)
        chosen = []
        chosen_set = set()
        
        for idx in indices:
            m = all_masks[idx]
            if m in chosen_set:
                continue
            conflict = False
            for i in range(len(chosen)):
                a = chosen[i]
                for j in range(i + 1, len(chosen)):
                    b = chosen[j]
                    ab = a & b
                    if (a & m) == ab and (b & m) == ab:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                chosen.append(m)
                chosen_set.add(m)
            if time.time() > deadline:
                break
        return chosen
    
    # Try multiple random greedy runs and keep the best
    best_family = []
    best_size = 0
    
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        family = greedy_random(seed)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local search: try to add more sets
    if time.time() < deadline - 10:
        family = best_family[:]
        family_set = set(family)
        added = True
        while added and time.time() < deadline - 5:
            added = False
            for mask in all_masks:
                if mask not in family_set:
                    conflict = False
                    for i in range(len(family)):
                        a = family[i]
                        for j in range(i + 1, len(family)):
                            b = family[j]
                            ab = a & b
                            if (a & mask) == ab and (b & mask) == ab:
                                conflict = True
                                break
                        if conflict:
                            break
                    if not conflict:
                        family.append(mask)
                        family_set.add(mask)
                        added = True
                        if len(family) > best_size:
                            best_size = len(family)
                            best_family = family[:]
    
    # Swap-based local search to potentially find larger families
    if time.time() < deadline - 5:
        family = best_family[:]
        family_set = set(family)
        current_size = len(family)
        
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            if not family:
                break
            
            idx_out = random.randint(0, len(family) - 1)
            mask_out = family[idx_out]
            
            # Find a random mask not in family
            attempts = 0
            while attempts < 100:
                mask_in = random.choice(all_masks)
                if mask_in not in family_set:
                    break
                attempts += 1
            else:
                continue
            
            # Check if mask_in can be added to family without mask_out
            fam_temp = family[:idx_out] + family[idx_out + 1:]
            conflict = False
            for i in range(len(fam_temp)):
                a = fam_temp[i]
                for j in range(i + 1, len(fam_temp)):
                    b = fam_temp[j]
                    ab = a & b
                    if (a & mask_in) == ab and (b & mask_in) == ab:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                new_family = fam_temp + [mask_in]
                if len(new_family) > current_size:
                    family = new_family
                    family_set.remove(mask_out)
                    family_set.add(mask_in)
                    current_size = len(family)
                    if current_size > best_size:
                        best_size = current_size
                        best_family = family[:]
    
    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
