import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, leaving 5s margin for safety
    
    max_val = 4096
    max_size = 80
    
    best_set = []
    best_size = 0
    
    def get_sums_for(x, current_set):
        """
        Compute the set of triple sums involving x, given current_set.
        current_set must NOT contain x.
        """
        sums = set()
        k = len(current_set)
        
        # 3x
        sums.add(3 * x)
        
        # 2x + y
        for y in current_set:
            sums.add(2 * x + y)
            
        # x + 2y
        for y in current_set:
            sums.add(x + 2 * y)
            
        # x + y + z
        for i in range(k):
            y = current_set[i]
            for j in range(i + 1, k):
                z = current_set[j]
                sums.add(x + y + z)
                
        return sums

    def can_add(x, current_set, sum_set):
        """
        Check if x can be added to current_set without creating duplicate triple sums.
        """
        if x in current_set:
            return False
        new_sums = get_sums_for(x, current_set)
        # Check for intersection
        for s in new_sums:
            if s in sum_set:
                return False
        return True

    def run_greedy(seed=None):
        """
        Runs a greedy algorithm to build a B_3 set.
        """
        current_set = []
        sum_set = set()
        
        candidates = list(range(1, max_val + 1))
        
        if seed is not None:
            random.seed(seed)
            random.shuffle(candidates)
        else:
            # Deterministic increasing order for initial base
            pass 

        for x in candidates:
            if time.time() > deadline:
                break
            
            if can_add(x, current_set, sum_set):
                new_sums = get_sums_for(x, current_set)
                current_set.append(x)
                sum_set.update(new_sums)
                
                if len(current_set) >= max_size:
                    break
        
        return current_set, sum_set

    def local_swap_optimize(s, sum_set, max_swaps=2000):
        """
        Optimizes the set by swapping elements to potentially allow more additions later.
        Returns the best set found and its sum_set.
        """
        if not s:
            return s, sum_set
        
        s = sorted(s)
        n = len(s)
        
        # Ensure sum_set is consistent with s
        # Rebuild sum_set to be safe if passed incorrectly
        sum_set = set()
        for x in s:
             # This is inefficient, but we assume sum_set is passed correctly from greedy
             # For safety in local_swap, we might want to rebuild if we are unsure.
             # But get_sums_for requires x NOT in set.
             # Let's just trust the passed sum_set from greedy.
             pass

        best_s = s[:]
        best_sum_set = sum_set.copy()
        
        # Simple local search: try to swap elements to reduce max value or variance
        # Or just random walk to find a state where addition is possible.
        
        # We will integrate this into the main SA loop instead of a separate function
        # to allow adding elements during the search.
        
        return best_s, best_sum_set

    # 1. Run deterministic increasing greedy
    set1, sums1 = run_greedy()
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1[:]
        
    # 2. Simulated Annealing / Random Walk to improve
    # State: current_set, current_sums
    # Objective: Maximize len(current_set)
    
    current_set = set1
    current_sums = sums1
    
    # If greedy didn't give us a good start, try a few random seeds
    if best_size < 10:
        for seed in range(10):
             s, sums = run_greedy(seed=seed)
             if len(s) > best_size:
                 best_size = len(s)
                 best_set = s[:]
                 current_set = s
                 current_sums = sums
             if time.time() > deadline - 5:
                 break

    # SA Loop
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute candidates for swaps? No, random selection is better.
    # We need to track which elements are in the set for fast checking
    set_elements = set(current_set)
    
    while time.time() < deadline - 2.0 and T > T_min:
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # Try to add a random element
        # Sample from 1..4096
        if random.random() < 0.3: # 30% chance to try add
            x = random.randint(1, max_val)
            if x not in set_elements:
                new_sums = get_sums_for(x, current_set)
                if not (new_sums & current_sums):
                    # Valid addition
                    current_set.append(x)
                    set_elements.add(x)
                    current_sums.update(new_sums)
                    # Reset temperature? Or just continue
                    # Adding is a big win, maybe restart T?
                    # T = 1.0 
                    continue
        
        # Try to swap
        if len(current_set) > 0:
            # Pick a random element to remove
            idx = random.randint(0, len(current_set) - 1)
            x_remove = current_set[idx]
            
            # Compute sums to remove
            # We need the set of sums involving x_remove
            # Note: current_set still contains x_remove
            # We need to compute sums of x_remove with (current_set - {x_remove})
            others = [y for y in current_set if y != x_remove]
            sums_to_remove = get_sums_for(x_remove, others)
            
            # Temporarily remove
            current_sums.difference_update(sums_to_remove)
            current_set.pop(idx)
            set_elements.discard(x_remove)
            
            # Pick a random element to add
            x_add = random.randint(1, max_val)
            if x_add not in set_elements:
                new_sums = get_sums_for(x_add, current_set)
                if not (new_sums & current_sums):
                    # Valid swap
                    current_set.append(x_add)
                    set_elements.add(x_add)
                    current_sums.update(new_sums)
                    
                    # Acceptance: Always accept valid swap of same size?
                    # Yes, to explore.
                    # We might want to prefer swaps that reduce max value?
                    # For now, random walk.
                else:
                    # Invalid swap, revert
                    current_set.append(x_remove)
                    set_elements.add(x_remove)
                    current_sums.update(sums_to_remove)
                    # Rejection
                    # Maybe cool down faster?
                    T *= cooling_rate
            else:
                # x_add already in set, revert
                current_set.append(x_remove)
                set_elements.add(x_remove)
                current_sums.update(sums_to_remove)
                T *= cooling_rate
        else:
            T *= cooling_rate
            
        T *= cooling_rate

    return best_set
