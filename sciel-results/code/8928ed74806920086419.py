import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    best_set = None
    best_size = 0
    
    # Initial deterministic run (descending)
    def run_greedy(order):
        selected = []
        reachable = {0}
        for x in order:
            new_sums = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected

    initial_order = list(range(100, 0, -1))
    best_set = run_greedy(initial_order)
    best_size = len(best_set)
    
    # Randomized search
    # We can try different orders.
    # Bias towards larger numbers might be good, but pure random explores more.
    # Let's mix: 50% descending, 50% random?
    # Or just random permutations.
    
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        
        # Optional: bias towards larger numbers?
        # Sorting by random key?
        # Let's just use shuffled order.
        
        current_set = run_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # If we found a set of size > 35, we might want to stop early?
        # Or keep searching for even better.
        # 35 is the known best from deterministic.
        # If we find 36, that's an improvement.
        
    return best_set
