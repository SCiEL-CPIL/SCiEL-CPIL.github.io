import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit (safe margin before 60s/90s kill)
    
    # --- Bitset-based Greedy Initialization ---
    # This is much faster than the set-based approach in the current program.
    # reachable is an integer where bit i is 1 if sum i is reachable.
    def greedy_solve(order):
        selected = []
        reachable = 1  # Bit 0 is set (sum 0)
        for x in order:
            # Shift reachable by x and OR with original
            # We only care about bits up to 200 (actually slightly more, but we check 200)
            # (reachable << x) creates new sums
            new_reach = reachable | (reachable << x)
            # Check if sum 200 is now reachable
            if not (new_reach >> 200) & 1:
                selected.append(x)
                reachable = new_reach
        return selected

    # --- Fixed Orders ---
    orders = [
        list(range(100, 0, -1)),  # Descending (Seed)
        list(range(1, 101)),      # Ascending
    ]
    
    # Center Out
    center_order = [50]
    for i in range(1, 51):
        if 50 + i <= 100:
            center_order.append(50 + i)
        if 50 - i >= 1:
            center_order.append(50 - i)
    orders.append(center_order)
    
    # Ends In
    ends_order = []
    for i in range(1, 51):
        ends_order.append(i)
        ends_order.append(101 - i)
    orders.append(ends_order)

    best_set = None
    best_size = -1

    # Evaluate fixed orders
    for order in orders:
        if time.time() > deadline:
            break
        res = greedy_solve(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    # If no set found (shouldn't happen), fallback
    if best_set is None:
        best_set = greedy_solve(list(range(100, 0, -1)))
        best_size = len(best_set)

    current_set = set(best_set)
    current_size = best_size
    all_elements = list(range(1, 101))
    
    # --- Fast Validity Check using Bitset ---
    def check_validity(s):
        r = 1
        for x in s:
            r |= (r << x)
            if (r >> 200) & 1:
                return False
        return True

    # --- Local Search / Simulated Annealing Hybrid ---
    # We combine the directed search (remove 1, add 2) with random perturbations.
    # We use bitsets for validity checks to maximize throughput.
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Strategy 1: Directed Improvement (Remove 1, Add 2)
        # This allows us to jump over local optima where single swaps fail.
        if current_size > 0 and random.random() < 0.7:
            remove_x = random.choice(list(current_set))
            candidates = [x for x in all_elements if x not in current_set]
            
            if len(candidates) >= 2:
                # Try a few random pairs
                found_improvement = False
                for _ in range(5): # Limit attempts per iteration for speed
                    if time.time() > deadline:
                        break
                    add_pair = random.sample(candidates, 2)
                    new_set = current_set - {remove_x}
                    new_set.update(add_pair)
                    
                    if check_validity(new_set):
                        new_size = len(new_set)
                        if new_size > current_size:
                            current_set = new_set
                            current_size = new_size
                            if current_size > best_size:
                                best_set = current_set.copy()
                                best_size = current_size
                            found_improvement = True
                            break
                        elif new_size == current_size and random.random() < 0.2:
                            # Plateau move
                            current_set = new_set
                            found_improvement = True
                            break
                
                if found_improvement:
                    continue # Restart main loop with new state

        # Strategy 2: Random Perturbation (Flip 1-2 bits)
        # Used to escape local minima or explore different regions.
        num_flips = random.randint(1, 2)
        new_set = current_set.copy()
        
        for _ in range(num_flips):
            x = random.choice(all_elements)
            if x in new_set:
                new_set.discard(x)
            else:
                new_set.add(x)
                
        if check_validity(new_set):
            new_size = len(new_set)
            
            if new_size > current_size:
                current_set = new_set
                current_size = new_size
                if current_size > best_size:
                    best_set = current_set.copy()
                    best_size = current_size
            elif new_size == current_size:
                if random.random() < 0.5:
                    current_set = new_set
            else:
                # Downhill move
                delta = current_size - new_size
                # Accept with probability exp(-delta/T)
                # Using log for stability if T is small
                if random.random() < pow(2.71828, -delta / T):
                    current_set = new_set
                    current_size = new_size
        else:
            # Invalid, reject (stay in current state)
            pass
            
        T *= cooling
        if T < 0.05:
            # Reheat
            T = 1.0
            # Optional: reset to best to avoid drifting too far
            if random.random() < 0.1:
                current_set = best_set.copy()
                current_size = best_size

    return list(best_set)
