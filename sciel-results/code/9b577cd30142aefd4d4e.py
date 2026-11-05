import random
import time

def _is_valid(set_ints):
    """
    Check if all subset sums of the given set are distinct.
    """
    n = len(set_ints)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += set_ints[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _get_max(set_ints):
    return max(set_ints)

def construct():
    n = 12
    deadline = time.time() + 85.0
    
    # Initial solution: Powers of 2
    best_set = [1 << i for i in range(n)]
    best_max = _get_max(best_set)
    
    # Simulated Annealing
    current_set = best_set.copy()
    current_max = best_max
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() < deadline and T > T_min:
        # Pick a random index to change
        i = random.randint(0, n - 1)
        old_val = current_set[i]
        
        # Generate a new value
        # We want to encourage smaller values
        # Let's pick a value from 1 to max(current_max, 2048)
        # But bias towards smaller values
        max_possible = max(current_max, 2048)
        # Use exponential distribution to bias towards small values?
        # Or just uniform in [1, max_possible]
        new_val = random.randint(1, max_possible)
        
        # Ensure distinctness
        if new_val == old_val:
            continue
        if new_val in current_set[:i] + current_set[i+1:]:
            continue
            
        current_set[i] = new_val
        
        # Check validity
        if _is_valid(current_set):
            new_max = _get_max(current_set)
            
            if new_max < current_max:
                # Always accept improvement
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_set = current_set.copy()
            else:
                # Accept with probability exp(-(new_max - current_max) / T)
                delta = new_max - current_max
                if delta == 0:
                    # Neutral move, accept with some probability to explore
                    if random.random() < 0.5:
                        current_max = new_max
                else:
                    if random.random() < pow(2.718281828, -delta / T):
                        current_max = new_max
                if current_max != new_max:
                    # Revert if not accepted
                    current_set[i] = old_val
        else:
            # Revert
            current_set[i] = old_val
            
        T *= cooling_rate
        
    return best_set
