import time
import random
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Precompute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    
    # Convert to integer weights
    # weight for s is L // s
    weights = {s: L // s for s in range(1, 23)}

    def is_dissociated_fast(S):
        """
        Checks if S (list of ints from 1..22) is dissociated.
        Uses integer arithmetic with LCM.
        Returns True if all subset sums are distinct.
        """
        if not S:
            return True

        w = [weights[s] for s in S]
        
        sums = {0}
        for val in w:
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    # Known good sets to seed the search
    # These are found via manual inspection or previous runs
    # Size 11 is a strong baseline.
    base_sets = [
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 21, 22],
        [1, 2, 4, 5, 9, 11, 13, 17, 19, 21, 22],
        [1, 3, 4, 5, 7, 11, 13, 17, 19, 21, 22],
        [1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 22]
    ]
    
    # Verify and filter base sets
    valid_base_sets = []
    for bs in base_sets:
        if is_dissociated_fast(bs):
            valid_base_sets.append(bs)
            
    if not valid_base_sets:
        # Fallback to a simple greedy if bases fail
        current = []
        for x in range(1, 23):
            if is_dissociated_fast(current + [x]):
                current.append(x)
        valid_base_sets.append(current)

    best_set = max(valid_base_sets, key=len)
    best_size = len(best_set)
    
    # If we already found a very large set (e.g., 13+), we can return early
    # but let's try to push for 14 or 15.
    
    # Simulated Annealing
    # State: set S
    # Objective: maximize |S|
    # Moves: Add, Remove, Swap
    
    # We will run SA for the remaining time
    # We use multiple restarts if time permits, or just one long run
    
    # To make SA effective, we should start from the best known set
    # and try to grow it.
    
    # However, simple greedy often gets stuck. SA with swaps can move around.
    
    # Let's define the SA parameters
    # T0: initial temperature
    # T_end: final temperature
    # alpha: cooling rate
    
    # We'll run a loop until deadline
    # We'll keep the best set found so far
    
    # Helper to get random element not in set
    def get_random_not_in(S):
        available = [x for x in range(1, 23) if x not in S]
        if not available:
            return None
        return random.choice(available)
    
    # Helper to get random element in set
    def get_random_in(S):
        if not S:
            return None
        return random.choice(S)
        
    # We'll try a few independent SA runs with different seeds
    # to diversify the search
    
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts if time.time() < deadline else 0
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize current set
        # Mix: sometimes start from best, sometimes from a random valid set
        if r == 0 or random.random() < 0.5:
            current_set = list(best_set)
        else:
            # Generate a random valid set
            # Start with empty and add random elements
            rand_set = []
            candidates = list(range(1, 23))
            random.shuffle(candidates)
            for x in candidates:
                if is_dissociated_fast(rand_set + [x]):
                    rand_set.append(x)
            current_set = rand_set
            
        current_size = len(current_set)
        
        # SA Parameters
        # We want to maximize size.
        # Energy = -size
        # Delta E = -Delta Size
        
        # We'll use a time-based cooling schedule
        start_t = time.time()
        # Estimate steps we can take
        # Each step involves a check which is O(2^k) where k is size.
        # Size ~ 12, so 2^12 = 4096 sums. Very fast.
        # We can do thousands of steps per second.
        
        T = 1.0
        T_min = 0.01
        alpha = 0.999  # Cooling rate per step
        
        # Local best tracking
        local_best_set = list(current_set)
        local_best_size = current_size
        
        # Loop until time for this restart is up
        while time.time() < start_t + time_per_restart:
            # Choose a move
            move_type = random.random()
            
            new_set = list(current_set)
            
            if move_type < 0.4:
                # Add
                if len(new_set) < 22:
                    x = get_random_not_in(new_set)
                    if x is not None:
                        new_set.append(x)
            elif move_type < 0.7:
                # Remove
                if len(new_set) > 0:
                    x = get_random_in(new_set)
                    if x is not None:
                        new_set.remove(x)
            else:
                # Swap
                if len(new_set) > 0 and len(new_set) < 22:
                    x_in = get_random_in(new_set)
                    x_out = get_random_not_in(new_set)
                    if x_in is not None and x_out is not None:
                        new_set.remove(x_in)
                        new_set.append(x_out)
            
            # Check validity
            if not is_dissociated_fast(new_set):
                # Invalid move, reject (or accept with very low prob, but let's just reject for simplicity as we want valid sets)
                # Actually, in SA, we usually require valid states or penalize invalid ones.
                # Since we need to return a valid set, let's stick to valid states.
                # If we reject all invalid moves, we might get stuck.
                # Better: If invalid, reject.
                T *= alpha
                continue
                
            new_size = len(new_set)
            
            # Calculate Delta E
            # We want to maximize size, so minimize E = -size
            delta_e = -new_size - (-current_size) # = current_size - new_size
            
            # Acceptance probability
            if delta_e < 0:
                # Improvement
                accept = True
            else:
                # Worse or same
                if T > 0:
                    prob = 1.0 / (1.0 + delta_e * 10) # Simple heuristic
                    # Standard SA: exp(-delta_e / T)
                    # Since delta_e is small integer, let's use exp
                    if delta_e == 0:
                        accept = random.random() < 0.5
                    else:
                        accept = random.random() < (2.718 ** (-delta_e / T))
                else:
                    accept = False
            
            if accept:
                current_set = new_set
                current_size = new_size
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_set = list(current_set)
                    
                    if local_best_size > best_size:
                        best_size = local_best_size
                        best_set = list(local_best_set)
                        
            T *= alpha

    # Final Polish: Try to add any missing elements to best_set
    missing = [x for x in range(1, 23) if x not in best_set]
    for x in missing:
        if time.time() > deadline:
            break
        new_set = best_set + [x]
        if is_dissociated_fast(new_set):
            best_set = new_set
            # Restart missing list
            missing = [x for x in range(1, 23) if x not in best_set]
            break

    # One more pass of random swaps to see if we can swap in something that allows addition
    # This is a bit complex, so we'll rely on the SA above.
    
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set

