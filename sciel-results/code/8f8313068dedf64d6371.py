import time
import random

def get_cost(x):
    max_disc = 0
    n = len(x)
    for d in range(1, n):
        for s in range(0, n - d + 1):
            curr = 0
            min_val = 0
            max_val = 0
            idx = s
            while idx < n:
                curr += x[idx]
                if curr < min_val:
                    min_val = curr
                if curr > max_val:
                    max_val = curr
                idx += d
            disc = max_val - min_val
            if disc > max_disc:
                max_disc = disc
                # Early exit if we hit a high value? 
                # If max_disc is already high, we might break early if we just want to compare?
                # But we need the exact value for the objective.
    return max_disc

def construct():
    n = 256
    start_time = time.time()
    time_limit = 50.0 # Leave 10s margin
    
    # Initialize
    # Use a random start, or maybe a greedy start?
    # Random is fine for SA.
    x = [random.choice([-1, 1]) for _ in range(n)]
    current_cost = get_cost(x)
    best_cost = current_cost
    best_x = x[:]
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    
    while time.time() - start_time < time_limit:
        # Pick random index
        i = random.randint(0, n - 1)
        # Flip
        old_val = x[i]
        x[i] = -x[i]
        
        new_cost = get_cost(x)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_x = x[:]
        else:
            # Accept with probability
            delta = new_cost - current_cost
            if delta > 0:
                prob = 1.0 / (1.0 + delta * (T + 0.1)) # Simple heuristic, or exp(-delta/T)
                # exp is better
                import math
                prob = math.exp(-delta / T)
                if random.random() > prob:
                    # Reject
                    x[i] = old_val
                # else accept
            else:
                # This shouldn't happen if logic is correct (new_cost < current_cost handled above)
                pass
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_x
