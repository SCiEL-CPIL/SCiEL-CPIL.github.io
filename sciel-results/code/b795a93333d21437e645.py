import random
import time
import math

def calc_max_discrepancy(signs):
    n = len(signs)
    max_abs = 0
    # Pre-computed structure isn't strictly necessary for N=256, 
    # but using local variables speeds up the loop.
    for d in range(1, n):
        for r in range(d):
            curr_sum = 0
            # Iterate through the AP
            for idx in range(r, n, d):
                curr_sum += signs[idx]
                if curr_sum > max_abs:
                    max_abs = curr_sum
                elif curr_sum < -max_abs:
                    max_abs = -curr_sum
    return max_abs

def construct():
    n = 256
    # Thue-Morse initialization
    # i -> parity of bit count
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    start_time = time.time()
    limit_time = start_time + 55.0 # 55 seconds to be safe
    
    current_max = calc_max_discrepancy(signs)
    best_signs = signs[:]
    best_max = current_max
    
    # Phase 1: Hill Climbing to find a strong local minimum
    # This is very fast and gets us to a good state quickly
    improved = True
    while improved and time.time() < limit_time - 30.0: # Leave 30s for SA
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            if time.time() > limit_time - 30.0:
                break
            old_sign = signs[i]
            signs[i] = -old_sign
            new_max = calc_max_discrepancy(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
                if new_max < best_max:
                    best_max = new_max
                    best_signs = signs[:]
            else:
                signs[i] = old_sign
    
    # Phase 2: Simulated Annealing to escape local minima
    # Start with a temperature high enough to accept some bad moves
    T = 5.0
    T_min = 0.1
    alpha = 0.995
    
    current_max = calc_max_discrepancy(signs) # Ensure we are at best_signs state? 
    # Actually, let's restart from best_signs for SA
    signs = best_signs[:]
    current_max = best_max
    
    while time.time() < limit_time:
        if T < T_min:
            break
            
        i = random.randint(0, n - 1)
        old_sign = signs[i]
        signs[i] = -old_sign
        
        new_max = calc_max_discrepancy(signs)
        
        delta = new_max - current_max
        
        if delta < 0:
            # Better, always accept
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs[:]
        else:
            # Worse, accept with probability exp(-delta / T)
            # If delta is large, probability is ~0
            # If delta is small, probability is significant
            if random.random() < math.exp(-delta / T):
                current_max = new_max
            else:
                # Reject
                signs[i] = old_sign
        
        T *= alpha
        
    return best_signs
