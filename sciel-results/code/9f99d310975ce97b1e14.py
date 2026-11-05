import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # 60s limit, leave 5s margin
    
    # 1. Greedy Initialization
    selected = []
    reachable = 1
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
            
    best_set = set(selected)
    best_size = len(best_set)
    
    # Current state
    current_set = best_set.copy()
    current_size = best_size
    
    # Precompute all elements
    all_elements = list(range(1, 101))
    
    # Helper to check validity using bitset with early exit
    def check_validity(s):
        r = 1
        for x in s:
            r |= (r << x)
            if (r >> 200) & 1:
                return False, r
        return True, r
    
    # Directed local search: try exchanges that increase size
    # Try to remove one element and add two elements (net +1)
    
    while time.time() < deadline:
        # Try to remove one element and add two elements
        if current_size == 0:
            break
        
        # Choose a random element to remove
        remove_x = random.choice(list(current_set))
        
        # Get candidates not in set
        candidates = [x for x in all_elements if x not in current_set]
        if len(candidates) < 2:
            break
        
        # Try a few random pairs
        found_improvement = False
        for _ in range(20):
            if time.time() > deadline:
                break
            add_pair = random.sample(candidates, 2)
            new_set = current_set - {remove_x}
            new_set.update(add_pair)
            
            valid, r = check_validity(new_set)
            if valid and len(new_set) > current_size:
                # Apply the change
                current_set = new_set
                current_size = len(new_set)
                if current_size > best_size:
                    best_set = current_set.copy()
                    best_size = current_size
                found_improvement = True
                break
        
        if not found_improvement:
            # Try a random perturbation: flip 1-2 bits
            num_flips = random.randint(1, 2)
            new_set = current_set.copy()
            for _ in range(num_flips):
                x = random.choice(all_elements)
                if x in new_set:
                    new_set.discard(x)
                else:
                    new_set.add(x)
            
            valid, r = check_validity(new_set)
            if valid:
                if len(new_set) > current_size:
                    current_set = new_set
                    current_size = len(new_set)
                    if current_size > best_size:
                        best_set = current_set.copy()
                        best_size = current_size
                elif len(new_set) == current_size and random.random() < 0.3:
                    # Plateau move
                    current_set = new_set

    return list(best_set)
