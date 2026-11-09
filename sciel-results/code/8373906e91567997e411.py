import random

def _greedy_solve(order):
    """
    Greedily constructs a B2[2] set from the given order of candidates.
    Returns the list of chosen integers.
    """
    chosen = []
    counts = {}
    
    for x in order:
        # Check if adding x violates the B2[2] condition
        valid = True
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                valid = False
                break
        
        if valid:
            s = 2 * x
            if counts.get(s, 0) >= 2:
                valid = False
        
        if valid:
            chosen.append(x)
            # Update the count dictionary
            for y in chosen[:-1]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            
            s = 2 * x
            counts[s] = counts.get(s, 0) + 1
            
    return chosen

def construct():
    # Base candidates
    candidates = list(range(1, 1001))
    
    best_set = []
    
    # 1. Try deterministic orders first (ascending, descending)
    deterministic_orders = [
        candidates,
        sorted(candidates, reverse=True)
    ]
    
    for order in deterministic_orders:
        current_set = _greedy_solve(order)
        if len(current_set) > len(best_set):
            best_set = current_set
            
    # 2. Try random restarts
    # We can afford a few thousand restarts within the time limit
    num_restarts = 2000
    
    for _ in range(num_restarts):
        order = candidates[:]
        random.shuffle(order)
        current_set = _greedy_solve(order)
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
