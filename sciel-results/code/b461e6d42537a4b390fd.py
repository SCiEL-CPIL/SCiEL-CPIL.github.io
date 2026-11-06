import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def can_add(x, reachable):
        return (200 - x) not in reachable

    def get_reachable(S):
        reachable = {0}
        for x in S:
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
        return reachable

    def greedy_solve(order):
        S = []
        reachable = {0}
        for x in order:
            if can_add(x, reachable):
                S.append(x)
                new_sums = {s + x for s in reachable if s + x <= 200}
                reachable |= new_sums
        return S, reachable

    # 1. Generate initial candidates
    best_S = []
    best_size = 0
    
    # Fixed orders
    fixed_orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(100, 0, -1, 2)) + list(range(99, 0, -1, 2)),
        list(range(1, 101, 2)) + list(range(100, 0, -1, 2)),
    ]
    
    for order in fixed_orders:
        if time.time() > deadline: break
        S, R = greedy_solve(order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)

    # Random initial orders
    for _ in range(20):
        if time.time() > deadline: break
        order = list(range(1, 101))
        random.shuffle(order)
        S, R = greedy_solve(order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)

    # 2. Local Search
    # We will try to improve best_S
    # Strategy: Pick a random element y to remove, then re-greedy the rest with a random order.
    # If we find a larger set, we accept it.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element in best_S
        # To avoid bias, we can shuffle the list of elements to remove
        candidates_to_remove = best_S[:]
        random.shuffle(candidates_to_remove)
        
        for y in candidates_to_remove:
            if time.time() > deadline: break
            
            # Remove y
            S_temp = [x for x in best_S if x != y]
            
            # Determine remaining elements
            S_temp_set = set(S_temp)
            remaining = [x for x in range(1, 101) if x not in S_temp_set]
            
            # Shuffle remaining to try different greedy paths
            random.shuffle(remaining)
            
            # Greedy insert
            S_new = S_temp[:]
            R_new = get_reachable(S_temp)
            
            for x in remaining:
                if can_add(x, R_new):
                    S_new.append(x)
                    new_sums = {s + x for s in R_new if s + x <= 200}
                    R_new |= new_sums
            
            if len(S_new) > best_size:
                best_S = S_new
                best_size = len(S_new)
                improved = True
                break # Restart loop with new best

    return best_S
