import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)
WEIGHTS = {x: L // x for x in UNIVERSE}

def get_conflicts(S):
    """
    Returns a list of elements x in S that are in conflict.
    x is in conflict if W_x is a subset sum of S\{x}.
    """
    conflicts = []
    S_list = list(S)
    n = len(S_list)
    if n == 0:
        return conflicts
    
    weights = [WEIGHTS[x] for x in S_list]
    
    for i, x in enumerate(S_list):
        target = weights[i]
        others = weights[:i] + weights[i+1:]
        
        if not others:
            continue
            
        m = len(others)
        mid = m // 2
        left = others[:mid]
        right = others[mid:]
        
        left_sums = {0}
        for w in left:
            left_sums = left_sums | {s + w for s in left_sums}
            
        found = False
        right_sums = {0}
        for w in right:
            new_sums = {s + w for s in right_sums}
            for s in new_sums:
                if target - s in left_sums:
                    found = True
                    break
            if found:
                break
            right_sums |= new_sums
            
        if found:
            conflicts.append(x)
            
    return conflicts

def is_valid(S):
    return len(get_conflicts(S)) == 0

def construct():
    deadline = time.time() + 85
    
    # Start with a known large valid set: {5, ..., 24}
    current = set(range(5, 25))
    # Verify it is valid
    while not is_valid(current):
        # If not valid, remove conflicts
        conflicts = get_conflicts(current)
        if not conflicts:
            break
        # Remove one conflict
        current.remove(random.choice(conflicts))
        
    best_size = len(current)
    best_set = set(current)
    
    # Simulated Annealing / Local Search
    # We try to add elements, and if stuck, do swaps or removals.
    
    removed = set(UNIVERSE) - current
    
    # Precompute subset sums for the current set? 
    # No, we recompute in is_valid.
    
    # To speed up, we can maintain the subset sums of current set?
    # But the set changes.
    
    # Let's use a simple loop.
    
    temp = time.time()
    no_improve_count = 0
    
    while time.time() < deadline:
        # 1. Try to add a random element from removed
        if removed:
            x = random.choice(list(removed))
            candidate = current | {x}
            if is_valid(candidate):
                current = candidate
                removed.remove(x)
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
                    no_improve_count = 0
                continue
        
        # 2. If no add possible, try swaps
        # Remove one from current, add one from removed
        # This keeps size same, but might enable future adds
        swapped = False
        if current and removed:
            # Try a few random swaps
            for _ in range(10):
                if time.time() > deadline:
                    break
                x_to_remove = random.choice(list(current))
                x_to_add = random.choice(list(removed))
                candidate = (current - {x_to_remove}) | {x_to_add}
                if is_valid(candidate):
                    # Accept swap with some probability (annealing)
                    # Since size is same, we only accept if it "looks better" or random
                    # For simplicity, accept if random() < 0.5
                    if random.random() < 0.5:
                        current = candidate
                        removed.remove(x_to_add)
                        removed.add(x_to_remove)
                        swapped = True
                        break
        
        if not swapped and not removed:
             # Stuck
             # Remove a random element to shake
             if current:
                 x_to_remove = random.choice(list(current))
                 current.remove(x_to_remove)
                 removed.add(x_to_remove)
                 no_improve_count += 1
                 
        # If we are stuck for a long time, reset to best and try different strategy?
        # Or just keep shaking.
        
    return list(best_set)
