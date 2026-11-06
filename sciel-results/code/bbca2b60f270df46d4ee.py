import random
import time
import math

def construct():
    n = 256
    all_nums = list(range(1, n+1))
    
    # Helper functions for speed
    def is_valid_add(present, x):
        # Check if adding x creates a 4-AP
        # 1. x is 4th: x-3d, x-2d, x-d
        for d in range(1, (x-1)//3 + 1):
            if present[x-d] and present[x-2*d] and present[x-3*d]:
                return False
        # 2. x is 3rd: x-2d, x-d, x+d
        for d in range(1, (x-1)//2 + 1):
            if present[x-d] and present[x-2*d] and present[x+d]:
                return False
        # 3. x is 2nd: x-d, x+d, x+2d
        for d in range(1, (n-x)//2 + 1):
            if present[x-d] and present[x+d] and present[x+2*d]:
                return False
        # 4. x is 1st: x+d, x+2d, x+3d
        for d in range(1, (n-x)//3 + 1):
            if present[x+d] and present[x+2*d] and present[x+3*d]:
                return False
        return True

    def calc_delta_add(present, x):
        delta = 0
        # x is 3rd: x-2d, x-d
        for d in range(1, (x-1)//2 + 1):
            if present[x-d] and present[x-2*d]:
                delta += 1
        # x is 2nd: x-d, x+d
        for d in range(1, min(x-1, n-x) + 1):
            if present[x-d] and present[x+d]:
                delta += 1
        # x is 1st: x+d, x+2d
        for d in range(1, (n-x)//2 + 1):
            if present[x+d] and present[x+2*d]:
                delta += 1
        return delta

    def calc_delta_remove(present, x):
        delta = 0
        # x is 3rd
        for d in range(1, (x-1)//2 + 1):
            if present[x-d] and present[x-2*d]:
                delta += 1
        # x is 2nd
        for d in range(1, min(x-1, n-x) + 1):
            if present[x-d] and present[x+d]:
                delta += 1
        # x is 1st
        for d in range(1, (n-x)//2 + 1):
            if present[x+d] and present[x+2*d]:
                delta += 1
        return delta

    def count_3aps(present):
        count = 0
        for a in range(1, n+1):
            if not present[a]: continue
            for d in range(1, (n-a)//2 + 1):
                if present[a+d] and present[a+2*d]:
                    count += 1
        return count

    # Initial state: Greedy construction
    present = [False] * (n+1)
    for x in range(1, n+1):
        if is_valid_add(present, x):
            present[x] = True
            
    current_score = count_3aps(present)
    best_score = current_score
    best_present = present[:]
    
    start_time = time.time()
    timeout = 55
    
    # Simulated Annealing
    T = 5.0 # Initial temperature
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        # Random move: Add or Remove
        # Bias towards adding if set is small, removing if large?
        # Just 50/50 for now.
        if random.random() < 0.5:
            # Add
            candidates = [x for x in all_nums if not present[x]]
            if not candidates: continue
            x = random.choice(candidates)
            if not is_valid_add(present, x):
                continue
            delta = calc_delta_add(present, x)
            new_score = current_score + delta
            # Accept if better, or with prob exp(-delta/T) if worse
            if delta >= 0 or random.random() < math.exp(-delta / T):
                present[x] = True
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_present = present[:]
        else:
            # Remove
            candidates = [x for x in all_nums if present[x]]
            if not candidates: continue
            x = random.choice(candidates)
            delta = -calc_delta_remove(present, x)
            new_score = current_score + delta
            # Removing reduces score usually, so delta is negative.
            # Accept if better (unlikely) or with prob exp(-delta/T)
            if delta >= 0 or random.random() < math.exp(-delta / T):
                present[x] = False
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_present = present[:]
                    
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor temperature

    return [x for x in all_nums if best_present[x]]
