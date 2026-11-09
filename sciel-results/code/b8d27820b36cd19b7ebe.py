import time
import random
import numpy as np
from itertools import combinations

N = 512
TIME_LIMIT = 75  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Precompute all 4-term arithmetic progressions in [1, 512]
    # A 4-AP is (a, a+d, a+2d, a+3d) where all terms are in [1, 512] and d > 0
    # We'll store these as lists of 4-tuples for quick lookup
    
    # For efficiency, we'll use a different approach:
    # For each pair (a, d), the 4-AP is (a, a+d, a+2d, a+3d)
    # a ranges from 1 to 512, d ranges from 1 to (512-a)//3
    
    # Let's build a constraint structure:
    # For each element x, store which 4-APs contain x
    # And for each 4-AP, store its 4 elements
    
    # Actually, let's use a smarter approach with numpy for the greedy + local search
    
    # First, let's try a randomized greedy with multiple restarts
    # and a local search (swap-and-fill) to improve
    
    best_set = set()
    best_size = 0
    
    # Precompute conflicts: for each element, what other elements it conflicts with
    # Two elements conflict if they appear together in some 4-AP
    # But actually, for 4-AP-free, we need to check: adding x creates a 4-AP
    # if there exist a, d such that {a, a+d, a+2d, a+3d} includes x and 3 others already in set
    
    # Let's use a different strategy:
    # 1. Generate a random ordering of [1, 512]
    # 2. Greedily add elements if they don't form a 4-AP with existing elements
    # 3. Local search: try to swap elements or add more
    
    def can_add(x, current_set):
        """Check if adding x to current_set creates a 4-term AP"""
        # For each d, check if x is part of a 4-AP with 3 elements in current_set
        # x could be at position 0, 1, 2, or 3 in the 4-AP
        
        # Position 0: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in current_set and (x + 2*d) in current_set and (x + 3*d) in current_set:
                return False
        
        # Position 1: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in current_set and (x + d) in current_set and (x + 2*d) in current_set:
                return False
        
        # Position 2: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (x - 2*d) in current_set and (x - d) in current_set and (x + d) in current_set:
                return False
        
        # Position 3: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d) in current_set and (x - 2*d) in current_set and (x - d) in current_set:
                return False
        
        return True
    
    def greedy_randomized():
        """Greedy with random ordering"""
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        current_set = set()
        for x in elements:
            if can_add(x, current_set):
                current_set.add(x)
        return current_set
    
    def local_search(current_set):
        """Try to improve the set by swapping or adding elements"""
        current_set = set(current_set)
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT:
            improved = False
            
            # Try adding elements not in the set
            for x in range(1, N + 1):
                if x not in current_set:
                    if can_add(x, current_set):
                        current_set.add(x)
                        improved = True
                        break
            
            if not improved:
                # Try swapping: remove one element, add another
                for remove_x in list(current_set):
                    if time.time() - start_time > TIME_LIMIT:
                        break
                    trial_set = current_set - {remove_x}
                    # Try to add multiple elements
                    added_any = False
                    for add_x in range(1, N + 1):
                        if add_x not in trial_set:
                            if can_add(add_x, trial_set):
                                trial_set.add(add_x)
                                added_any = True
                    if len(trial_set) > len(current_set):
                        current_set = trial_set
                        improved = True
                        break
        
        return current_set
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        current_set = greedy_randomized()
        current_set = local_search(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # If we have time, try a few more targeted improvements
    if time.time() - start_time < TIME_LIMIT - 2:
        # Try a different initialization: start with a known good structure
        # Use the base-3 method (Salem-Spencer type) as a starting point
        # Numbers with only 0s and 1s in base 3, scaled appropriately
        base3_set = set()
        for i in range(N + 1):
            # Convert to base 3, check if all digits are 0 or 1
            n = i
            valid = True
            while n > 0:
                if n % 3 == 2:
                    valid = False
                    break
                n //= 3
            if valid:
                base3_set.add(i)
        
        # Filter to [1, 512]
        base3_set = {x for x in base3_set if 1 <= x <= 512}
        
        if len(base3_set) > best_size:
            best_set = base3_set
            best_size = len(base3_set)
        
        # Local search on this
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(improved_set)
    
    return sorted(best_set)
