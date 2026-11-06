"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _can_add(selected, x):
    """Check if x can be added to selected without violating the condition."""
    n = len(selected)
    if n < 3:
        # x cannot divide sum of 3 others if there are fewer than 3 others
        # For existing a, can a divide sum of 3 others including x?
        # If n=2, others for a (if a is one of the 2) = {x, other}. Size 2. No triples.
        # If n=2, for the other element a', others = {x, a}. Size 2. No triples.
        # So no checks needed when n < 3.
        return True
    
    # Check 1: Does x divide sum of any 3 distinct in selected?
    for t in combinations(selected, 3):
        if sum(t) % x == 0:
            return False
    
    # Check 2: Does any a in selected divide sum of 3 distinct in selected U {x} excluding a?
    # Only need to check triples including x: {x, b, c} with b,c in selected\{a}, b!=c.
    for a in selected:
        # For each a, check if there exist b, c in selected (b!=a, c!=a, b!=c) such that (x+b+c) % a == 0
        # Precompute: for fixed a and x, we need (b+c) % a == (a - x%a) % a
        target = (a - x % a) % a
        # Collect values of b mod a for b in selected, b != a
        mods = []
        for b in selected:
            if b != a:
                mods.append(b % a)
        # Now check if any pair of mods sums to target mod a
        # Use set for O(1) lookup
        seen = set()
        for m in mods:
            needed = (target - m) % a
            if needed in seen:
                return False
            seen.add(m)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin before 90s kill
    
    best_selected = []
    best_size = 0
    
    def update_best(sel):
        nonlocal best_selected, best_size
        if len(sel) > best_size:
            best_selected = sel[:]
            best_size = len(best_selected)
    
    # Strategy 1: Descending (high to low)
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    update_best(selected)
    
    # Strategy 2: Only large numbers (>96) descending
    selected = []
    for x in range(192, 96, -1):
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    update_best(selected)
    
    # Strategy 3: Numbers in [65, 192] descending
    selected = []
    for x in range(192, 64, -1):
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    update_best(selected)
    
    # Strategy 4: Numbers in [48, 192] descending
    selected = []
    for x in range(192, 47, -1):
        if time.time() > deadline:
            break
        if _can_add(selected, x):
            selected.append(x)
    update_best(selected)
    
    # Strategy 5: Randomized greedy
    num_random_runs = 0
    while time.time() < deadline - 10:
        if time.time() > deadline:
            break
        
        selected = []
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if _can_add(selected, x):
                selected.append(x)
        
        update_best(selected)
        num_random_runs += 1
        
        if best_size == 48:
            break
    
    # Local Search on the best found set
    if best_size > 0 and time.time() < deadline:
        current = best_selected[:]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try removing one element and adding two
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                reduced = current[:i] + current[i+1:]
                reduced_set = set(reduced)
                
                # Try adding elements not in reduced, from high to low
                added_count = 0
                new_current = reduced[:]
                new_set = reduced_set.copy()
                
                for x in range(192, 0, -1):
                    if time.time() > deadline:
                        break
                    if x in new_set:
                        continue
                    if _can_add(new_current, x):
                        new_current.append(x)
                        new_set.add(x)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 2:
                    # Size increased by 1 (removed 1, added 2)
                    if len(new_current) > best_size:
                        best_selected = new_current[:]
                        best_size = len(best_selected)
                        current = new_current[:]
                        improved = True
                        break
    
    return best_selected
# EVOLVE-BLOCK-END
