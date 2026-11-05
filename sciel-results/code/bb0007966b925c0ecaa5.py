import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0 # 90s limit, leave 5s margin
    
    # 1. Greedy Initialization (Same logic as seed, but using bitset for speed)
    selected = []
    reachable = 1 # Bit i set means sum i is reachable
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
            
    best_set = set(selected)
    best_size = len(best_set)
    
    # If the greedy solution is already very good, we still search.
    # Local Search
    current_set = best_set.copy()
    current_size = best_size
    
    # Precompute all elements
    all_elements = list(range(1, 101))
    
    # Helper to check validity using bitset
    def is_valid_and_size(s):
        r = 1
        # Optimization: sort? No, order doesn't matter for final set.
        # But early exit if 200 reached.
        for x in s:
            r |= (r << x)
            if (r >> 200) & 1:
                return -1, False
        return len(s), True
        
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    
    # To avoid getting stuck in a local optimum of the greedy solution,
    # we might want to perturb more aggressively at the start.
    
    while time.time() < deadline:
        # Generate neighbor
        # Flip 1 or 2 bits
        num_flips = random.randint(1, 2)
        new_set = current_set.copy()
        
        for _ in range(num_flips):
            x = random.choice(all_elements)
            if x in new_set:
                new_set.remove(x)
            else:
                new_set.add(x)
                
        new_size, is_valid = is_valid_and_size(new_set)
        
        if is_valid:
            if new_size > current_size:
                current_set = new_set
                current_size = new_size
                if new_size > best_size:
                    best_set = new_set
                    best_size = new_size
            elif new_size == current_size:
                # Plateau move
                if random.random() < 0.5:
                    current_set = new_set
            else:
                # Downhill move
                delta = current_size - new_size
                # Accept with probability exp(-delta/T)
                # Using log for stability if T is small, but delta is small (1 or 2)
                if random.random() < pow(2.71828, -delta / T):
                    current_set = new_set
                    current_size = new_size
        # else invalid, reject
        
        T *= cooling
        if T < 0.01:
            # Reheat or reset to best if stuck?
            # If we reheat, we might escape local minima.
            T = 1.0
            # Optional: reset current to best with some probability
            if random.random() < 0.1:
                current_set = best_set.copy()
                current_size = best_size

    return list(best_set)
