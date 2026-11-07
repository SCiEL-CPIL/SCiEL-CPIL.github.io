import time
import random

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _smallest_multiple_at_least(d, L):
    return ((L + d - 1) // d) * d

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    # 1. Greedy Scan
    candidates = set()
    for d in DIVISORS:
        candidates.update(range(d, MAX_VAL + 1, d))
    
    best_span = float('inf')
    best_reps = None

    D = DIVISORS
    
    # Iterate candidates
    # To make it faster, we can sample if the set is too large, but 600k is small.
    for L in candidates:
        if time.time() > deadline - 5.0: # Reserve 5s for local search
            break
            
        reps = []
        valid = True
        for d in D:
            m = _smallest_multiple_at_least(d, L)
            if m > MAX_VAL:
                valid = False
                break
            reps.append(m)
        
        if not valid:
            continue
            
        if len(set(reps)) != N:
            continue
            
        span = max(reps) - min(reps)
        
        if span < best_span:
            best_span = span
            best_reps = reps[:]
            if best_span <= 9:
                break

    if best_reps is None:
        return [int(x) for x in DIVISORS]

    # 2. Local Search
    # Start from best_reps
    current_reps = best_reps[:]
    current_span = best_span
    
    random.seed(42)
    
    # Simple Hill Climbing with some random restarts if stuck?
    # Or just random moves.
    
    # We define a move: pick index i, change reps[i] by +/- d[i]
    # Check validity: 1 <= new_val <= MAX_VAL, distinctness.
    # Check improvement: new_span < current_span.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        i = random.randint(0, N - 1)
        d = D[i]
        direction = random.choice([-1, 1])
        new_val = current_reps[i] + direction * d
        
        if new_val < 1 or new_val > MAX_VAL:
            continue
            
        # Check distinctness
        # Create a set of all reps, remove old, add new
        # Since N=10, this is fast
        old_val = current_reps[i]
        # If new_val is already in the set (excluding old_val), it's a collision
        # We can just build the new list and check set size
        new_reps = current_reps[:]
        new_reps[i] = new_val
        
        if len(set(new_reps)) != N:
            continue
            
        new_span = max(new_reps) - min(new_reps)
        
        if new_span < current_span:
            current_reps = new_reps
            current_span = new_span
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a random restart from a different greedy solution?
                # Or just break if we've tried enough
                # Given the greedy scan was thorough, local search is just polish.
                # If we can't improve, we might be in a local min.
                # But greedy solutions are global minima of the "greedy" class.
                # Non-greedy improvements are likely small.
                break
                
    if current_span < best_span:
        best_span = current_span
        best_reps = current_reps

    return [int(x) for x in best_reps]
# EVOLVE-BLOCK-END
