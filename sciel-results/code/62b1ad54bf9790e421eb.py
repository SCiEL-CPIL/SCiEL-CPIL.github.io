import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety within 120s limit
    
    prescribed = [1, 2, 5, 11]
    fixed_set = frozenset(prescribed)
    
    def is_valid_sidon(s):
        """Check if set s forms a valid Sidon set (all pairwise sums distinct)."""
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                s_val = s[i] + s[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    def count_conflicts(s):
        """Count the number of duplicate sums (for SA objective)."""
        sum_counts = defaultdict(int)
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                s_val = s[i] + s[j]
                sum_counts[s_val] += 1
        conflicts = 0
        for c in sum_counts.values():
            if c > 1:
                conflicts += 1
        return conflicts
    
    # 1. Greedy Construction to get a strong baseline
    a = list(prescribed)
    sums = set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
            
    best_set = a
    best_size = len(best_set)
    
    if best_size >= 20:
        return best_set
    
    # 2. Advanced Simulated Annealing with Multi-Start and Restart Strategy
    # We will try to find a valid Sidon set of size 20 first.
    # If that fails within time, we'll try 19, etc.
    
    def attempt_find(target_size, time_limit):
        """
        Try to find a valid Sidon set of target_size using Simulated Annealing.
        Returns a valid set if found, else None.
        """
        end_time = time.time() + time_limit
        if end_time > deadline:
            end_time = deadline - 1.0
            
        if time.time() >= end_time:
            return None
            
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
                
            # Initialize candidate set
            # Start with prescribed elements
            current = list(prescribed)
            remaining_needed = target_size - len(current)
            
            # Fill with random distinct elements from [1, 284]
            pool = [x for x in range(1, 285) if x not in current]
            random.shuffle(pool)
            current.extend(pool[:remaining_needed])
            
            random.shuffle(current)
            
            # Initial state
            current_conflicts = count_conflicts(current)
            
            if current_conflicts == 0:
                return current
                
            # SA Parameters
            temp = 10.0
            min_temp = 0.001
            cooling_rate = 0.995
            max_steps = 5000
            stagnation_limit = 1000
            stagnation_counter = 0
            
            best_local_conflicts = current_conflicts
            best_local_set = current[:]
            
            for step in range(max_steps):
                if time.time() > end_time:
                    return None
                if temp < min_temp:
                    break
                    
                # Pick a random index to change
                idx = random.randint(0, len(current) - 1)
                
                # Do not change prescribed elements
                if current[idx] in fixed_set:
                    continue
                    
                old_val = current[idx]
                new_val = random.randint(1, 284)
                
                if new_val == old_val:
                    continue
                
                # Check if new_val is already in the set
                if new_val in current:
                    continue
                    
                # Calculate delta conflicts efficiently
                # We recompute full conflicts for simplicity and correctness, 
                # though O(N^2) per step is heavy for N=20. 
                # 20*19/2 = 190 pairs. 190 ops per step. 
                # 5000 steps * 190 = 950k ops. Very fast.
                
                # Temporarily swap
                current[idx] = new_val
                new_conflicts = count_conflicts(current)
                
                delta = new_conflicts - current_conflicts
                
                if new_conflicts < best_local_conflicts:
                    best_local_conflicts = new_conflicts
                    best_local_set = current[:]
                    
                if new_conflicts == 0:
                    return current
                
                # Acceptance
                if delta <= 0:
                    current_conflicts = new_conflicts
                    stagnation_counter = 0
                else:
                    prob = math.exp(-delta / temp)
                    if random.random() < prob:
                        current_conflicts = new_conflicts
                        stagnation_counter = 0
                    else:
                        # Reject
                        current[idx] = old_val
                        stagnation_counter += 1
                        
                temp *= cooling_rate
                
                if stagnation_counter > stagnation_limit:
                    break
                    
            # If we ended this restart with a better local best, update global best
            # (We don't really track a global best across restarts in this function, 
            #  but we can return the best_local_set if it has 0 conflicts, which is handled above)
            
            # If we didn't find 0 conflicts, we continue to next restart.
            # Optionally, we could seed the next restart with best_local_set.
            # Let's do that to help convergence.
            if best_local_conflicts < count_conflicts(current):
                 current = best_local_set[:]
                 current_conflicts = best_local_conflicts

        return None
    
    # Allocate time for target sizes 20 down to best_size + 1
    targets = [20, 19, 18, 17, 16, 15, 14]
    # Dynamic time allocation: give more time to higher targets
    # Total available ~115s - greedy_time (negligible)
    # Let's try 20 with 60s, 19 with 30s, etc.
    time_allocs = [60, 30, 15, 5, 2, 1, 1]
    
    for i, target in enumerate(targets):
        if target <= best_size:
            continue
            
        alloc = time_allocs[i]
        # Ensure we don't exceed deadline
        if time.time() + alloc > deadline:
            alloc = max(0.1, deadline - time.time() - 1)
            
        if alloc < 0.5:
            break
            
        res = attempt_find(target, alloc)
        
        if res is not None:
            best_set = res
            best_size = len(res)
            break
            
    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
