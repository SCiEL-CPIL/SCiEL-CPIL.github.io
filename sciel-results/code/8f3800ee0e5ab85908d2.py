import random
import time
import math

def calc_max_disc(signs, n=256, limit=None):
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            curr = 0
            min_p = 0
            max_p = 0
            idx = r
            while idx < n:
                curr += signs[idx]
                if curr < min_p: min_p = curr
                if curr > max_p: max_p = curr
                idx += d
            disc = max_p - min_p
            if disc > max_disc:
                max_disc = disc
                if limit is not None and max_disc > limit:
                    return max_disc
    return max_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    # Initial solution
    signs = [random.choice((-1, 1)) for _ in range(n)]
    current_disc = calc_max_disc(signs)
    best_disc = current_disc
    best_signs = signs[:]
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = random.randint(0, n - 1)
        signs[i] *= -1
        
        # Calculate new discrepancy with a limit to speed up rejection
        # If the new discrepancy is much worse, we don't need the exact value
        limit = current_disc + int(T * 5) + 1
        new_disc = calc_max_disc(signs, n, limit)
        
        delta = new_disc - current_disc
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            # Reject
            signs[i] *= -1
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_signs
