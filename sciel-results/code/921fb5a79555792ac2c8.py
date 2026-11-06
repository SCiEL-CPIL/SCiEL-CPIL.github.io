import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 1..16, so bits 1..16
    candidates = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
        
    max_size = 0
    best_family = []
    
    # Run multiple randomized greedy + local search attempts
    while time.time() < deadline:
        random.shuffle(candidates)
        family = []
        family_set = set()
        
        # Phase 1: Greedy Construction
        for mask in candidates:
            if is_valid_fast(family, mask):
                family.append(mask)
                family_set.add(mask)
        
        if len(family) > max_size:
            max_size = len(family)
            best_family = family.copy()
            
        # Phase 2: Local Search (Swapping)
        # Try to improve the family by swapping elements
        # We track steps without size increase to decide when to restart
        steps_without_improvement = 0
        local_best_size = len(family)
        
        while time.time() < deadline:
            # If we have run out of time or are stuck, break
            if steps_without_improvement > 2000:
                break
                
            # Pick a random candidate from the complement
            # To avoid picking an element already in the family, we can
            # just pick and check. The probability of picking an existing one
            # is K/N. If K is small, this is fine. If K is large, it's less efficient.
            # Better: Pick from a list of candidates not in family?
            # Maintaining that list is expensive.
            # Given N=1820 and K~100, picking random is okay.
            
            idx = random.randint(0, len(candidates) - 1)
            c = candidates[idx]
            
            if c in family_set:
                continue
            
            valid, blockers = is_valid_with_blockers(family, c)
            
            if valid:
                family.append(c)
                family_set.add(c)
                if len(family) > max_size:
                    max_size = len(family)
                    best_family = family.copy()
                steps_without_improvement = 0
                local_best_size = len(family)
            else:
                # Conflict found
                # Randomly pick one blocker to remove
                a, b = blockers
                if random.random() < 0.5:
                    victim = a
                else:
                    victim = b
                
                # Remove victim
                family.remove(victim)
                family_set.remove(victim)
                
                # Try to add c
                # We must re-check validity because there might be other blockers
                valid2, _ = is_valid_fast(family, c)
                
                if valid2:
                    family.append(c)
                    family_set.add(c)
                    # Size is same (removed 1, added 1)
                    # This is a swap. It doesn't increase size.
                    # We increment steps_without_improvement because size didn't increase.
                    steps_without_improvement += 1
                else:
                    # Swap failed. Restore victim.
                    family.append(victim)
                    family_set.add(victim)
                    steps_without_improvement += 1

    # Convert best_family masks back to sorted lists of integers
    result = []
    for m in best_family:
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))
        
    return result

def is_valid_fast(family, mask):
    """
    Check if mask can be added to family without forming a sunflower.
    Returns True if valid, False otherwise.
    """
    n = len(family)
    for i in range(n):
        a = family[i]
        for j in range(i + 1, n):
            b = family[j]
            core = a & b
            if (core & mask) == core:
                # Check disjointness from A\K and B\K
                # A\K = a ^ core
                # B\K = b ^ core
                if (mask & (a ^ core)) == 0 and (mask & (b ^ core)) == 0:
                    return False
    return True

def is_valid_with_blockers(family, mask):
    """
    Check if mask can be added to family.
    Returns (True, None) if valid.
    Returns (False, (a, b)) if invalid, where a, b are a blocking pair.
    """
    n = len(family)
    for i in range(n):
        a = family[i]
        for j in range(i + 1, n):
            b = family[j]
            core = a & b
            if (core & mask) == core:
                if (mask & (a ^ core)) == 0 and (mask & (b ^ core)) == 0:
                    return False, (a, b)
    return True, None
