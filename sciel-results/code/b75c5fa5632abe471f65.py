import time
import random
import numpy as np
from collections import defaultdict

N = 256

# EVOLVE-BLOCK-START

def _is_valid_solution(arr):
    """
    Verifies:
    1. All elements are distinct integers in [1, N].
    2. It is a Sidon set (all unordered pair sums a+b with a<=b are distinct).
    3. It is maximal (adding any missing element from [1, N] destroys the Sidon property).
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
    
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute all sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # Try to add x
        # New sums are {x+a for a in arr} U {2x}
        # If any of these are in `sums`, or if there's a conflict among them, x cannot be added.
        # If x CAN be added (no conflicts), the set is NOT maximal.
        
        # Check conflicts with existing sums
        conflict = False
        for a in arr:
            if (x + a) in sums:
                conflict = True
                break
        if not conflict:
            if (2 * x) in sums:
                conflict = True
        
        if not conflict:
            # x can be added, so the set is not maximal.
            return False
        
    return True

def _get_sums(arr):
    """Returns set of sums or None if not Sidon."""
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return None
            sums.add(s)
    return sums

def _is_maximal(arr, sums):
    """Checks if arr is maximal given its sums."""
    arr_set = set(arr)
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        conflict = False
        for a in arr:
            if (x + a) in sums:
                conflict = True
                break
        if not conflict and (2 * x) in sums:
            conflict = True
        if not conflict:
            return False
    return True

def _repair_to_maximal(sidon_set, rng, deadline):
    """
    Takes a Sidon set and makes it maximal by adding elements.
    Uses a heuristic to choose which elements to add to minimize final size.
    """
    arr = sorted(sidon_set)
    sums = _get_sums(arr)
    if sums is None:
        return None
    
    arr_set = set(arr)
    
    while not _is_maximal(arr, sums):
        if time.time() > deadline:
            return None
            
        # Find all candidates that can be added
        candidates = []
        for x in range(1, N + 1):
            if x in arr_set:
                continue
            conflict = False
            for a in arr:
                if (x + a) in sums:
                    conflict = True
                    break
            if conflict:
                continue
            if (2 * x) in sums:
                continue
            candidates.append(x)
            
        if not candidates:
            # Should not happen if not maximal, but safety
            return None
            
        # Heuristic: Pick the candidate that blocks the most other candidates.
        # If we add x, it creates new sums {x+a, 2x}.
        # These new sums will block any y such that y+a' is in new_sums or 2y is in new_sums.
        
        best_x = None
        max_blocked = -1
        
        # To save time, we might not check all candidates if there are many.
        # But N=256, so it's small.
        
        for x in candidates:
            # Simulate adding x
            new_sums = sums.copy()
            for a in arr:
                s = x + a
                if s in new_sums:
                    # This shouldn't happen as x is a candidate
                    break
                new_sums.add(s)
            new_sums.add(2 * x)
            
            # Count how many other candidates become invalid
            blocked = 0
            for y in candidates:
                if y == x:
                    continue
                # Check if y is blocked by new_sums
                y_conflict = False
                for a in arr:
                    if (y + a) in new_sums:
                        y_conflict = True
                        break
                if not y_conflict and (2 * y) in new_sums:
                    y_conflict = True
                if y_conflict:
                    blocked += 1
            
            if blocked > max_blocked:
                max_blocked = blocked
                best_x = x
                
        if best_x is None:
            best_x = rng.choice(candidates)
            
        # Add best_x
        arr.append(best_x)
        arr_set.add(best_x)
        for a in arr:
            sums.add(best_x + a)
        sums.add(2 * best_x)
        
    return arr

def _greedy_construct(order, rng=None):
    """
    Greedily constructs a Sidon set by iterating through `order`.
    """
    if order is None:
        order = list(range(1, N + 1))
        if rng:
            rng.shuffle(order)
        else:
            random.shuffle(order)
            
    selected = []
    sums = set()
    
    for x in order:
        conflict = False
        for a in selected:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
            
        if (2 * x) in sums:
            continue
            
        selected.append(x)
        for a in selected:
            sums.add(x + a)
        sums.add(2 * x)
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = []
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # 1. Generate initial population of diverse maximal Sidon sets
    population = []
    
    # Deterministic seeds
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
    ]
    
    for order in orders:
        s = _greedy_construct(order)
        # Repair to maximal
        repaired = _repair_to_maximal(s, rng, deadline)
        if repaired and _is_valid_solution(repaired):
            population.append(repaired)
            if len(repaired) < best_size:
                best_size = len(repaired)
                best_set = repaired[:]
                
    # Random restarts for diversity
    while time.time() < deadline - 10.0:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = _greedy_construct(order, rng)
        repaired = _repair_to_maximal(s, rng, deadline)
        if repaired and _is_valid_solution(repaired):
            population.append(repaired)
            if len(repaired) < best_size:
                best_size = len(repaired)
                best_set = repaired[:]
            if len(population) >= 20:
                break

    if not population:
        # Fallback
        s = _greedy_construct(list(range(N, 0, -1)))
        repaired = _repair_to_maximal(s, rng, deadline)
        if repaired:
            return repaired
        return s

    # 2. Genetic Algorithm / Crossover Loop
    # We keep the best few solutions in the population
    max_pop_size = 10
    
    while time.time() < deadline:
        # Select parents
        if len(population) < 2:
            # Need more diversity
            order = list(range(1, N + 1))
            rng.shuffle(order)
            s = _greedy_construct(order, rng)
            repaired = _repair_to_maximal(s, rng, deadline)
            if repaired:
                population.append(repaired)
            continue
            
        p1_idx = rng.randrange(len(population))
        p2_idx = rng.randrange(len(population))
        while p2_idx == p1_idx and len(population) > 1:
            p2_idx = rng.randrange(len(population))
            
        p1 = population[p1_idx]
        p2 = population[p2_idx]
        
        # Crossover: Create a random subset of the union of p1 and p2
        union_set = set(p1) | set(p2)
        child_raw = []
        for x in union_set:
            # Include with probability 0.5? Or 0.7?
            # If we include too many, Sidon repair will remove many.
            # If we include too few, Maximal repair will add many.
            # Let's try 0.5
            if rng.random() < 0.5:
                child_raw.append(x)
                
        # Repair to Sidon
        # Sort to make it deterministic-ish or random? Random is better for exploration.
        rng.shuffle(child_raw)
        child_sidon = []
        sums = set()
        for x in child_raw:
            conflict = False
            for a in child_sidon:
                if (x + a) in sums:
                    conflict = True
                    break
            if conflict:
                continue
            if (2 * x) in sums:
                continue
            child_sidon.append(x)
            for a in child_sidon:
                sums.add(x + a)
            sums.add(2 * x)
            
        # Repair to Maximal
        child_maximal = _repair_to_maximal(child_sidon, rng, deadline)
        
        if child_maximal and _is_valid_solution(child_maximal):
            new_size = len(child_maximal)
            if new_size < best_size:
                best_size = new_size
                best_set = child_maximal[:]
                
            # Update population
            # Keep top max_pop_size solutions
            population.append(child_maximal)
            population.sort(key=lambda x: len(x))
            if len(population) > max_pop_size:
                population = population[:max_pop_size]
                
            # Early exit if we hit a very small size
            if best_size <= 10:
                break

    if not _is_valid_solution(best_set):
        # Fallback to a known valid set
        s = _greedy_construct(list(range(N, 0, -1)))
        repaired = _repair_to_maximal(s, rng, deadline)
        if repaired:
            return repaired
        return s
        
    return best_set

# EVOLVE-BLOCK-END
