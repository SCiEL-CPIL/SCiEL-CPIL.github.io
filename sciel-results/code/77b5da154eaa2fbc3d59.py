import time
import random
import math

def _score(S):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d in S.
    Uses bitmask for fast checking.
    S is a set of integers in 1..512.
    """
    mask = 0
    for x in S:
        mask |= (1 << x)
    
    valid_ds = 0
    # d can be at most (512-1)/2 = 255.5, so max d is 255.
    # Actually, if a=1, a+2d=512, 2d=511, d=255.5 not int.
    # Max a=1, max a+2d=512. 2d <= 511. d <= 255.
    for d in range(1, 256):
        # Check if there exists a such that a, a+d, a+2d are in S
        # Bit a in mask is 1
        # Bit a in (mask >> d) is 1 iff bit a+d in mask is 1
        # Bit a in (mask >> 2d) is 1 iff bit a+2d in mask is 1
        # So we check if mask & (mask >> d) & (mask >> 2d) != 0
        if mask & (mask >> d) & (mask >> 2*d):
            valid_ds += 1
    return valid_ds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_set = None
    best_score = 0
    
    # Initial best: Random or simple AP
    # AP [1, 2, ..., 32] has d=1 only? 
    # 1,2,3 -> d=1. 1,3,5 -> d=1? No, 1,3,5 is d=1? No, 1,3,5 is 1, 1+2, 1+4. 
    # AP 1, 2, ..., 32. 
    # d=1: 1,2,3 yes.
    # d=2: 1,3,5 yes.
    # ...
    # d=15: 1,16,31 yes.
    # d=16: 1,17,33 (33 not in set).
    # So AP 1..32 has score 15.
    # Let's start with this as a strong baseline.
    initial_set = set(range(1, 33))
    best_set = set(initial_set)
    best_score = _score(best_set)
    
    # Simulated Annealing
    # We will run multiple SA runs with different restarts
    while time.time() < deadline:
        # Start SA with a random set or perturbed best
        # To explore, let's start with a random set sometimes, or best set
        if random.random() < 0.2:
            current_set = set(random.sample(range(1, 513), 32))
        else:
            current_set = set(best_set)
            
        current_score = _score(current_set)
        
        T = 5.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.999
        # Max steps per run, limited by time
        run_deadline = time.time() + 10.0
        
        while time.time() < run_deadline and T > T_min:
            # Pick a move: remove one, add one
            # To ensure distinctness, we pick an element to remove, then a new one not in set
            elems = list(current_set)
            elem_to_remove = random.choice(elems)
            
            # Pick new value
            # Random from 1..512
            new_val = random.randint(1, 512)
            if new_val in current_set:
                # Retry or skip? Skipping is fine, just wastes a bit of time
                # Better to pick from available?
                # For speed, random is fine, collision prob 32/512 ~ 6%
                continue
                
            new_set = set(current_set)
            new_set.remove(elem_to_remove)
            new_set.add(new_val)
            
            new_score = _score(new_set)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_set = new_set
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
                    
            T *= cooling_rate
            
    return list(best_set)
