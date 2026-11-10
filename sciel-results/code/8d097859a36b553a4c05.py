import time
from math import gcd
from functools import reduce
from itertools import combinations
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def _subset_sums_exists(target, values):
    """
    Check if target can be formed as a subset sum of values.
    Uses meet-in-the-middle for efficiency if n is large enough, otherwise simple DP.
    """
    if not values:
        return target == 0
    if target == 0:
        return True
    if target < 0:
        return False
    
    n = len(values)
    if n <= 20:
        # Simple DP with set of sums
        reachable = {0}
        for v in values:
            if v > target:
                continue
            new_sums = set()
            for s in reachable:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            reachable.update(new_sums)
            if target in reachable:
                return True
        return False
    else:
        # Meet in the middle
        half = n // 2
        left_vals = values[:half]
        right_vals = values[half:]
        
        left_sums = {0}
        for v in left_vals:
            if v > target:
                continue
            new_sums = set()
            for s in left_sums:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            left_sums.update(new_sums)
            
        right_sums = {0}
        for v in right_vals:
            if v > target:
                continue
            new_sums = set()
            for s in right_sums:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            right_sums.update(new_sums)
            
        # Check if any left + right == target
        for l in left_sums:
            if target - l in right_sums:
                return True
        return False

def _check_validity(S):
    """
    Check if S is a valid reciprocal-subsum-free set.
    Returns True if valid, False otherwise.
    """
    S_list = sorted(S)
    n = len(S_list)
    if n == 0:
        return True
        
    # Precompute LCM of all elements in S
    # This might overflow or be huge, but for [1..24] it's manageable.
    # LCM(1..24) is 5354228880.
    lcm_all = reduce(_lcm, S_list, 1)
    
    for x in S_list:
        target = lcm_all // x
        # Candidates are y in S such that y != x.
        # Since 1/x = sum(1/y), we need sum(lcm_all/y) = target.
        # Note: lcm_all/y is an integer.
        candidates = []
        for y in S_list:
            if y == x:
                continue
            candidates.append(lcm_all // y)
            
        if _subset_sums_exists(target, candidates):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Initialize with a random valid set or greedy
    # Let's start with a random subset of size 12
    all_elements = list(range(1, 25))
    random.seed(42)
    
    # Try to find a good initial set
    # Greedy construction: add elements if they don't break validity
    # Or just start with a random set and repair/optimize.
    
    # Strategy: Start with a random set of size 10-12.
    # Then use Simulated Annealing to swap elements to increase size or maintain size while improving "quality" (though objective is just size).
    # Since objective is just size, we can just try to add elements.
    # If we can't add, we swap.
    
    # Let's use a randomized local search.
    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start with a random subset
        current_set = set(random.sample(all_elements, 10))
        
        # Ensure it's valid? If not, remove elements until valid.
        while not _check_validity(current_set):
            # Remove a random element
            # Better: remove the one causing conflict?
            # For simplicity, random removal
            elem_to_remove = random.choice(list(current_set))
            current_set.remove(elem_to_remove)
            if not current_set:
                break
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
        # Simulated Annealing / Local Search
        # Objective: Maximize size.
        # Moves:
        # 1. Add a random element from complement.
        # 2. If invalid, remove a random element from current set (to restore validity) or reject.
        # 3. Swap: Remove one, add one.
        
        # We want to maximize size. So we prefer moves that increase size.
        # If size is same, we might accept to escape local minima?
        # But if size is same, the set is just different.
        # Let's try to increase size.
        
        current_size = len(current_set)
        temperature = 0.1
        cooling_rate = 0.95
        max_steps = 500
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if not current_set:
                break
                
            # Try to add a random element
            if len(current_set) < 24:
                candidates_to_add = [x for x in all_elements if x not in current_set]
                if candidates_to_add:
                    x = random.choice(candidates_to_add)
                    new_set = set(current_set)
                    new_set.add(x)
                    
                    if _check_validity(new_set):
                        # Accept
                        current_set = new_set
                        current_size += 1
                        if current_size > best_size:
                            best_set = set(current_set)
                            best_size = current_size
                        # Reset temperature or keep it?
                        # Since we improved, we can keep searching for more improvements
                        temperature = 0.1 
                        continue
                    else:
                        # Reject, try swap?
                        pass
            
            # Try swap: remove one, add one
            # This maintains size but might allow adding a "better" element later?
            # Actually, if we can't add, swapping might allow adding a different element later.
            
            if len(current_set) < 24 and candidates_to_add:
                # Pick a random element to remove
                x_to_remove = random.choice(list(current_set))
                x_to_add = random.choice(candidates_to_add)
                
                new_set = set(current_set)
                new_set.remove(x_to_remove)
                new_set.add(x_to_add)
                
                if _check_validity(new_set):
                    # Accept swap (size same, but different set)
                    # We accept with some probability or always if it helps future adds?
                    # Let's accept if it's "better" in some heuristic?
                    # Or just accept with probability exp(-1/T) to explore?
                    # Since size is same, we don't gain directly.
                    # But it might unblock an add.
                    # Let's accept with high probability to explore.
                    if random.random() < 0.5:
                        current_set = new_set
                        # Update candidates_to_add
                        candidates_to_add = [x for x in all_elements if x not in current_set]
                    
            temperature *= cooling_rate

    # Final local search on best_set to see if we can add more
    # Sort candidates by "safety" (larger numbers are safer to add?)
    # Actually, let's just try all possible additions.
    current_set = set(best_set)
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        for x in sorted(all_elements, reverse=True): # Try larger ones first? Or smaller?
            if x in current_set:
                continue
            new_set = set(current_set)
            new_set.add(x)
            if _check_validity(new_set):
                current_set = new_set
                changed = True
                break
        # If no change in this pass, try again? No, if we added one, we restart.
        # If we didn't add any, we are stuck.
        
    # One more pass with random order
    if time.time() < deadline:
        candidates = [x for x in all_elements if x not in current_set]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                break
            new_set = set(current_set)
            new_set.add(x)
            if _check_validity(new_set):
                current_set = new_set

    return list(current_set)

# EVOLVE-BLOCK-END
