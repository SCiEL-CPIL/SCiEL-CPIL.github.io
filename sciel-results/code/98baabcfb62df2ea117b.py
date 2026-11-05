import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    required = {1, 2, 5, 11}
    max_val = 284
    all_elements = set(range(1, max_val + 1))
    
    def get_cost(S):
        n = len(S)
        if n == 0:
            return 0
        S_list = list(S)
        sums = set()
        collisions = 0
        for i in range(n):
            ai = S_list[i]
            for j in range(i, n):
                s = ai + S_list[j]
                if s in sums:
                    collisions += 1
                else:
                    sums.add(s)
        return 100 * collisions - n

    # Initial solution: Greedy construction
    S = set(required)
    
    # Greedy fill: try to add elements that keep the set valid
    # We try elements in random order to avoid bias towards small numbers
    candidates = list(all_elements - S)
    random.shuffle(candidates)
    
    for x in candidates:
        new_S = S | {x}
        # Only accept if it improves the cost (i.e., remains valid and increases size)
        # If S is valid (cost = -n), new_S valid (cost = -n-1) is better.
        # If new_S invalid, cost is high, so not better.
        if get_cost(new_S) < get_cost(S):
            S = new_S
            # If we reached max size 20, we can stop early
            if len(S) == 20:
                return sorted(list(S))

    best_S = set(S)
    best_cost = get_cost(S)
    
    current_S = set(S)
    current_cost = get_cost(S)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995 # Slow cooling to explore more
    
    # Track stagnation
    stagnation_counter = 0
    max_stagnation = 500
    
    while T > T_min and time.time() < deadline:
        # Generate neighbor
        move_type = random.random()
        neighbor_S = set(current_S)
        
        if move_type < 0.4:
            # Add
            candidates = list(all_elements - neighbor_S)
            if candidates:
                x = random.choice(candidates)
                neighbor_S.add(x)
        elif move_type < 0.7:
            # Remove
            removable = [x for x in current_S if x not in required]
            if removable:
                x = random.choice(removable)
                neighbor_S.discard(x)
        else:
            # Swap
            removable = [x for x in current_S if x not in required]
            candidates = list(all_elements - current_S)
            if removable and candidates:
                x = random.choice(removable)
                y = random.choice(candidates)
                neighbor_S.discard(x)
                neighbor_S.add(y)
        
        neighbor_cost = get_cost(neighbor_S)
        
        # Acceptance criterion
        delta = neighbor_cost - current_cost
        if delta < 0:
            # Better solution
            current_S = neighbor_S
            current_cost = neighbor_cost
            stagnation_counter = 0
            if current_cost < best_cost:
                best_S = set(current_S)
                best_cost = current_cost
                # If we found a valid set of size 20, it's optimal
                if best_cost == -20:
                    break
        else:
            # Worse solution
            if random.random() < math.exp(-delta / T):
                current_S = neighbor_S
                current_cost = neighbor_cost
                stagnation_counter += 1
            else:
                stagnation_counter += 1
        
        # If stuck, kick the system
        if stagnation_counter > max_stagnation:
            # Randomly remove 2-5 non-required elements and add 2-5 random ones
            removable = [x for x in current_S if x not in required]
            if len(removable) > 2:
                kick_size = random.randint(2, 5)
                to_remove = random.sample(removable, min(kick_size, len(removable)))
                for x in to_remove:
                    current_S.discard(x)
                
                candidates = list(all_elements - current_S)
                if candidates:
                    to_add = random.sample(candidates, min(kick_size, len(candidates)))
                    for x in to_add:
                        current_S.add(x)
                current_cost = get_cost(current_S)
                stagnation_counter = 0
        
        T *= cooling_rate

    # Return the best set found
    return sorted(list(best_S))
