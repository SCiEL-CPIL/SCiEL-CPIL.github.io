import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# Precompute LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = _L * i // math.gcd(_L, i)

# Precompute values V[k] = L/k
_V = [0] * 25
for i in range(1, 25):
    _V[i] = _L // i

def _can_form(target, items):
    """
    Check if target can be formed as a sum of a subset of items.
    Uses meet-in-the-middle.
    """
    if target == 0:
        return True
    if not items:
        return target == 0
    
    # Pruning: if sum(items) < target, return False
    if sum(items) < target:
        return False
        
    n = len(items)
    mid = n // 2
    left_items = items[:mid]
    right_items = items[mid:]
    
    # Generate subset sums for left
    left_sums = {0}
    for item in left_items:
        new_sums = {s + item for s in left_sums}
        left_sums.update(new_sums)
        
    # Check if target is in left_sums
    if target in left_sums:
        return True
        
    # Generate subset sums for right
    right_sums = {0}
    for item in right_items:
        new_sums = {s + item for s in right_sums}
        right_sums.update(new_sums)
        
    # Check if there's a pair
    # Iterate over the smaller set for efficiency
    if len(left_sums) < len(right_sums):
        for l in left_sums:
            if (target - l) in right_sums:
                return True
    else:
        for r in right_sums:
            if (target - r) in left_sums:
                return True
                
    return False

def _is_valid(S):
    """
    Check if set S is reciprocal-subsum-free.
    """
    for x in S:
        target = _V[x]
        others = [_V[y] for y in S if y != x]
        if _can_form(target, others):
            return False
    return True

def construct():
    """
    Construct a large reciprocal-subsum-free subset of [1, 24].
    """
    # 1. Greedy Initialization
    # Try adding from large to small
    best_set = []
    for candidate in range(24, 0, -1):
        trial = best_set + [candidate]
        if _is_valid(trial):
            best_set = trial
            
    # Also try small to large?
    trial_small = []
    for candidate in range(1, 25):
        t = trial_small + [candidate]
        if _is_valid(t):
            trial_small = t
    if len(trial_small) > len(best_set):
        best_set = trial_small
        
    # 2. Local Search
    # State: best_set
    # Moves: 
    #   a) Add an element not in best_set
    #   b) Swap an element in best_set with one not in best_set
    
    current_set = list(best_set)
    best_size = len(current_set)
    best_set_found = list(current_set)
    
    # Precompute list of all elements
    all_elements = list(range(1, 25))
    
    # Cache for validity? 
    # Since sets are small, we can just re-check.
    
    # Try to improve size first
    improved = True
    while improved and _time_remaining() > 5.0:
        improved = False
        # Try adding any missing element
        for candidate in all_elements:
            if candidate in current_set:
                continue
            trial = current_set + [candidate]
            if _is_valid(trial):
                current_set = trial
                improved = True
                best_size = len(current_set)
                best_set_found = list(current_set)
                break
        
    # Now try swaps to potentially find a larger set later or just explore
    # We will do a random walk / simulated annealing style search
    # Objective: Maximize size. If same size, accept with prob 0.5 to escape local optima.
    
    random.seed(42)
    current_size = len(current_set)
    
    # We will run this until time runs out
    while _time_remaining() > 1.0:
        # Choose a move
        # 1. If possible, try to increase size
        # 2. Otherwise, swap
        
        # Generate random swap
        if current_set:
            u_idx = random.randint(0, len(current_set) - 1)
            u = current_set[u_idx]
            
            # Pick v not in current_set
            missing = [x for x in all_elements if x not in current_set]
            if not missing:
                break # Maxed out size 24? Unlikely to be valid, but possible
            
            v = random.choice(missing)
            
            # Construct new set
            new_set = list(current_set)
            new_set[u_idx] = v
            
            # Check validity
            if _is_valid(new_set):
                new_size = len(new_set)
                # Accept if size is same (random walk) or larger (improvement)
                # Since we only swap, size is same.
                # We accept to explore.
                current_set = new_set
                
                # If we found a set of same size, we keep it.
                # We don't update best_set_found unless size increases, 
                # but since we only swap, size doesn't increase here.
                # We need to try to increase size after swaps.
                
        # After some swaps, try to add elements again
        # To keep it simple, we just do swaps and then a greedy add pass
        # But greedy add is O(N^2) checks.
        
        # Let's do a greedy add pass every 10 swaps
        # This is a heuristic to ensure we don't miss an easy addition
        
        # Actually, let's just do:
        # 1. Random Swap
        # 2. Greedy Add Pass
        # If Greedy Add Pass increases size, update best.
        
        # To avoid doing Greedy Add Pass every time (which is expensive),
        # we can just do it when we accept a swap?
        # No, Greedy Add Pass is: for each missing x, if valid, add.
        # This is O(N) validity checks.
        
        # Let's structure the loop:
        # 1. Try to add any missing element (Greedy). If success, update best.
        # 2. If no addition possible, do a random swap.
        # 3. Repeat.
        
        # The current loop above does swaps. Let's add the greedy pass inside the loop.
        
        # Refactored Loop Logic:
        # We will just do:
        # while time:
        #   added = False
        #   for x in missing:
        #     if valid(current + x):
        #       current = current + x
        #       added = True
        #       update best
        #       break
        #   if not added:
        #     do random swap
        #     if valid(new):
        #       current = new
        
        # This is better.
        
        # Let's restart the loop with this logic
        break # Break out to use the better logic below

    # Re-implement search with better logic
    current_set = list(best_set_found)
    
    while _time_remaining() > 2.0:
        added = False
        missing = [x for x in all_elements if x not in current_set]
        
        # Shuffle missing to try different additions
        random.shuffle(missing)
        
        for x in missing:
            trial = current_set + [x]
            if _is_valid(trial):
                current_set = trial
                added = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set_found = list(current_set)
                break
        
        if not added:
            # Try a swap
            if current_set:
                u_idx = random.randint(0, len(current_set) - 1)
                u = current_set[u_idx]
                
                if missing:
                    v = random.choice(missing)
                    new_set = list(current_set)
                    new_set[u_idx] = v
                    if _is_valid(new_set):
                        current_set = new_set
                        # After swap, we might be able to add more, 
                        # which will be caught in the next iteration's 'missing' check.
                        # We don't update best_size here because size is same.
    
    return sorted(best_set_found)

