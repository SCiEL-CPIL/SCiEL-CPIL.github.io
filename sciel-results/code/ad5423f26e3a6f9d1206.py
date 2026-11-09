import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s/90s limit
    
    n = 16
    k = 4
    max_size = 250
    
    # Generate all possible 4-element subsets of [1..16]
    all_sets = [set(c) for c in combinations(range(1, n + 1), k)]
    total = len(all_sets)  # 1820
    
    # Precompute intersection matrix for quick sunflower checks
    # Two sets form a "pair intersection" - we need to check if adding a new set
    # creates a sunflower with any existing pair
    # For sets A, B, C to be a sunflower: A∩B = A∩C = B∩C
    
    # Strategy: greedy with local search / hill climbing
    # Start with a random greedy solution, then try to improve via swaps
    
    def is_sunflower_pair(i, j, family):
        """Check if set i and set j can form a sunflower with any set in family."""
        inter_ij = all_sets[i] & all_sets[j]
        for s in family:
            if s == all_sets[i] or s == all_sets[j]:
                continue
            if (all_sets[i] & s) == inter_ij and (all_sets[j] & s) == inter_ij:
                return True
        return False
    
    def count_sunflowers(family):
        """Count number of sunflower triples in family."""
        count = 0
        family_list = list(family)
        for a_idx, b_idx in combinations(range(len(family_list)), 2):
            a = family_list[a_idx]
            b = family_list[b_idx]
            inter_ab = a & b
            for c_idx in range(b_idx + 1, len(family_list)):
                c = family_list[c_idx]
                if (a & c) == inter_ab and (b & c) == inter_ab:
                    count += 1
        return count
    
    def is_valid_addition(new_set, family):
        """Check if adding new_set to family creates any sunflower."""
        for a, b in combinations(family, 2):
            inter_ab = a & b
            if (a & new_set) == inter_ab and (b & new_set) == inter_ab:
                return False
        return True
    
    def greedy_build():
        """Build a large sunflower-free family using greedy approach."""
        remaining = list(range(total))
        random.shuffle(remaining)
        family = []
        
        for idx in remaining:
            if len(family) >= max_size:
                break
            s = all_sets[idx]
            if is_valid_addition(s, family):
                family.append(s)
        
        return family
    
    # Try multiple random greedy builds and pick the best
    best_family = None
    best_size = 0
    attempts = 0
    max_attempts = 20000  # Will be limited by time
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5.0:
            break
        family = greedy_build()
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            attempts = attempt
    
    # Local search: try to swap out a set for a better one
    if best_family is not None:
        current = best_family[:]
        current_set = set(id(s) for s in current)
        
        for iteration in range(100000):
            if time.time() > deadline - 5.0:
                break
            if len(current) >= max_size:
                break
            
            # Try to find a set not in current that can be added
            # Or try to remove one set and add two
            if len(current) < max_size:
                # Try random additions
                for _ in range(50):
                    idx = random.randint(0, total - 1)
                    s = all_sets[idx]
                    if s in current:
                        continue
                    if is_valid_addition(s, current):
                        current.append(s)
                        break
            
            # Try 1-for-1 swaps to potentially enable more additions
            if len(current) < max_size:
                for _ in range(100):
                    if time.time() > deadline - 5.0:
                        break
                    # Pick a random set to remove
                    if not current:
                        break
                    rem_idx = random.randint(0, len(current) - 1)
                    removed = current[rem_idx]
                    current_without = [x for x in current if x is not removed]
                    
                    # Try to add two sets
                    added_any = False
                    for _add in range(2):
                        for _ in range(200):
                            idx = random.randint(0, total - 1)
                            s = all_sets[idx]
                            if s in current_without:
                                continue
                            if is_valid_addition(s, current_without):
                                current_without.append(s)
                                added_any = True
                                break
                        if len(current_without) >= max_size:
                            break
                    
                    if len(current_without) > len(current):
                        current = current_without
                        if len(current) > best_size:
                            best_size = len(current)
                            best_family = current[:]
        
        if best_family is None or len(best_family) < len(current):
            best_family = current
    
    # Final verification
    if best_family is None:
        # Fallback: simple greedy
        family = []
        for idx in range(total):
            if len(family) >= max_size:
                break
            if is_valid_addition(all_sets[idx], family):
                family.append(all_sets[idx])
        best_family = family
    
    # Verify no sunflowers
    verified = []
    for s in best_family:
        if is_valid_addition(s, verified):
            verified.append(s)
    
    return [sorted(s) for s in verified]
