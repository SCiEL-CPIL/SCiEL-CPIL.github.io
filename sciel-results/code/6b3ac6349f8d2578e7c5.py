import time
import random
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Precompute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    weights = {s: L // s for s in range(1, 23)}
    
    # For faster access in loops, use a list indexed by value
    weights_list = [0] * 23
    for i in range(1, 23):
        weights_list[i] = weights[i]

    def is_dissociated(S):
        """
        Checks if S (list of ints from 1..22) is dissociated.
        Uses early termination on collision detection.
        """
        if not S:
            return True
        
        sums = [0]
        wl = weights_list
        for s in S:
            w = wl[s]
            # Generate new sums by adding w to all existing sums
            new_sums = [x + w for x in sums]
            
            # Check for collisions
            # If the combined set has fewer elements than the sum of lengths,
            # it means there are duplicates.
            # len(sums) == len(new_sums)
            # We expect 2 * len(sums) unique elements.
            
            # Optimization: Check set size
            # Creating the set is the main cost.
            combined = sums + new_sums
            if len(set(combined)) < len(combined):
                return False
            
            sums = combined
            
        return True

    best_set = []
    best_size = 0

    # Strategy 1: Randomized Greedy Construction
    restarts = 0
    while time.time() < deadline - 20:
        restarts += 1
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        
        current_set = []
        for x in cand_order:
            new_set = current_set + [x]
            if is_dissociated(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Early exit if we found a very large set
        if best_size >= 14:
            break

    # Strategy 2: Local Search to improve best_set
    if best_set:
        current_set = list(best_set)
        current_size = len(current_set)
        stall_count = 0
        max_stall = 100
        
        while time.time() < deadline - 5 and stall_count < max_stall:
            improved = False
            
            # Try to add any missing element directly
            missing = [x for x in range(1, 23) if x not in current_set]
            for x in missing:
                if time.time() > deadline - 5: break
                new_set = current_set + [x]
                if is_dissociated(new_set):
                    current_set = new_set
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                stall_count = 0
                continue
            
            # Try swaps: Remove 1, Add 2 (Net +1 size)
            found_swap = False
            # To avoid O(N^3) overhead in Python, we limit the number of checks
            # or rely on the speed of is_dissociated.
            for i in range(len(current_set)):
                if time.time() > deadline - 5: break
                
                u = current_set[i]
                base_set = [x for j, x in enumerate(current_set) if j != i]
                
                # Try adding pairs from missing
                for a_idx in range(len(missing)):
                    if time.time() > deadline - 5: break
                    v = missing[a_idx]
                    for b_idx in range(a_idx + 1, len(missing)):
                        w = missing[b_idx]
                        new_set = base_set + [v, w]
                        if is_dissociated(new_set):
                            current_set = new_set
                            current_size += 1
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    break
            
            if improved:
                stall_count = 0
                missing = [x for x in range(1, 23) if x not in current_set]
            else:
                stall_count += 1
                # Random perturbation to escape local minima
                if missing and current_set:
                    u_idx = random.randint(0, len(current_set)-1)
                    u = current_set[u_idx]
                    v = random.choice(missing)
                    new_set = [x for j, x in enumerate(current_set) if j != u_idx] + [v]
                    if is_dissociated(new_set):
                        current_set = new_set
                        # Size unchanged, but configuration changed
                        # Don't reset stall count, but this might help future swaps
                        pass

    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
