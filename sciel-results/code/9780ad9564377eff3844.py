import time
import numpy as np
from itertools import combinations
import random
import math

# EVOLVE-BLOCK-START
# Use a large prime for modular arithmetic to detect collisions efficiently
# P must be > 2^22 to avoid wrap-around issues with subset sums
P = 2**23 + 5  # 8388613, a prime

# Precompute modular weights: w_i = P * i^{-1} mod P = P/i mod P
# Actually, we want weights such that subset sums are distinct mod P
# w_i = P * inv(i) mod P, but P*inv(i) mod P = 0 for all i
# Instead, use w_i = (P // i) * (P % i != 0 ? 1 : 0) ... this doesn't work either

# Better approach: use LCM scaling as before, but optimize the search
_LCM_22 = 1
for _i in range(1, 23):
    _LCM_22 = math.lcm(_LCM_22, _i)

_WEIGHTS = {}
for _i in range(1, 23):
    _WEIGHTS[_i] = _LCM_22 // _i

def _compute_sums(element_set):
    """Compute all subset sums for a given set of elements using their weights."""
    sums = [0]
    for e in element_set:
        w = _WEIGHTS[e]
        new_sums = [s + w for s in sums]
        sums.extend(new_sums)
    return sums

def _is_dissociated(element_set):
    """Check if all subset sums are distinct."""
    sums = _compute_sums(element_set)
    return len(sums) == len(set(sums))

def _count_collisions(element_set):
    """Count the number of duplicate subset sums (0 means dissociated)."""
    sums = _compute_sums(element_set)
    return len(sums) - len(set(sums))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    best_set = []
    best_size = 0
    
    # Phase 1: Systematic greedy with multiple random orderings
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Randomize the order of elements to try
        elements = list(range(1, 23))
        random.shuffle(elements)
        
        current_set = set()
        current_sums = [0]
        seen_sums = {0}
        
        # Greedy phase: try to add elements from the randomized list
        for e in elements:
            if e in current_set:
                continue
            w = _WEIGHTS[e]
            valid = True
            new_sums = []
            for s in current_sums:
                ns = s + w
                if ns in seen_sums:
                    valid = False
                    break
                new_sums.append(ns)
            if valid:
                current_set.add(e)
                current_sums.extend(new_sums)
                seen_sums.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Phase 2: Local search with simulated annealing-style swaps
        # Try to improve by removing one element and adding another
        improved = True
        max_swap_attempts = 100
        swap_attempts = 0
        
        while improved and swap_attempts < max_swap_attempts and time.time() < deadline:
            improved = False
            swap_attempts += 1
            
            # Try to add any element not in the set
            added = False
            for e in range(1, 23):
                if e in current_set:
                    continue
                w = _WEIGHTS[e]
                valid = True
                new_sums = []
                for s in current_sums:
                    ns = s + w
                    if ns in seen_sums:
                        valid = False
                        break
                    new_sums.append(ns)
                if valid:
                    current_set.add(e)
                    current_sums.extend(new_sums)
                    seen_sums.update(new_sums)
                    improved = True
                    added = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    break
            
            if added:
                continue
            
            # If no addition possible, try a random swap
            if not current_set:
                break
                
            remove_e = random.choice(list(current_set))
            temp_set = current_set - {remove_e}
            
            # Rebuild sums for temp_set
            temp_sums = [0]
            temp_seen = {0}
            for e in temp_set:
                w = _WEIGHTS[e]
                new_sums = [s + w for s in temp_sums]
                temp_sums.extend(new_sums)
                temp_seen.update(new_sums)
            
            # Try to add a different element to temp_set
            swapped = False
            candidates = [e for e in range(1, 23) if e not in temp_set and e != remove_e]
            random.shuffle(candidates)
            
            for add_e in candidates:
                w = _WEIGHTS[add_e]
                valid = True
                new_sums = []
                for s in temp_sums:
                    ns = s + w
                    if ns in temp_seen:
                        valid = False
                        break
                    new_sums.append(ns)
                if valid:
                    # Perform the swap
                    current_set = temp_set
                    current_set.add(add_e)
                    current_sums = temp_sums
                    current_sums.extend(new_sums)
                    seen_sums = temp_seen
                    seen_sums.update(new_sums)
                    
                    improved = True
                    swapped = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    break
            
            if not swapped:
                # Try a 2-swap: remove two, add two
                if len(current_set) >= 2:
                    rem1, rem2 = random.sample(list(current_set), 2)
                    temp_set2 = current_set - {rem1, rem2}
                    
                    temp_sums2 = [0]
                    temp_seen2 = {0}
                    for e in temp_set2:
                        w = _WEIGHTS[e]
                        new_sums2 = [s + w for s in temp_sums2]
                        temp_sums2.extend(new_sums2)
                        temp_seen2.update(new_sums2)
                    
                    add_candidates = [e for e in range(1, 23) if e not in temp_set2 and e not in (rem1, rem2)]
                    random.shuffle(add_candidates)
                    
                    # Try pairs of additions
                    for i in range(min(20, len(add_candidates))):
                        for j in range(i+1, min(20, len(add_candidates))):
                            add1, add2 = add_candidates[i], add_candidates[j]
                            w1 = _WEIGHTS[add1]
                            w2 = _WEIGHTS[add2]
                            
                            # Check if adding both is valid
                            valid = True
                            new_sums_a = []
                            for s in temp_sums2:
                                ns = s + w1
                                if ns in temp_seen2:
                                    valid = False
                                    break
                                new_sums_a.append(ns)
                            
                            if valid:
                                combined_sums = temp_sums2 + new_sums_a
                                combined_seen = set(temp_seen2) | set(new_sums_a)
                                
                                new_sums_b = []
                                for s in combined_sums:
                                    ns = s + w2
                                    if ns in combined_seen:
                                        valid = False
                                        break
                                    new_sums_b.append(ns)
                            
                            if valid:
                                current_set = temp_set2
                                current_set.add(add1)
                                current_set.add(add2)
                                current_sums = temp_sums2 + new_sums_a + new_sums_b
                                seen_sums = set(current_sums)
                                
                                improved = True
                                swapped = True
                                if len(current_set) > best_size:
                                    best_size = len(current_set)
                                    best_set = list(current_set)
                                break
                        if swapped:
                            break
                    if swapped:
                        break

    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
