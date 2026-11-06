"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    n = 256
    
    # Function to check if discrepancy <= K
    # Returns None if valid, else the first violation found (d, r, peak_idx)
    def find_violation(a, K):
        for d in range(1, n + 1):
            for r in range(d):
                cur_sum = 0
                # We only need to check up to the point where it might exceed K
                # But APs are short.
                for i in range(r, n, d):
                    cur_sum += a[i]
                    if cur_sum > K or cur_sum < -K:
                        return (d, r, i)
        return None

    # Calculate exact max discrepancy (for final verification or fallback)
    def calc_max_disc(a):
        max_abs = 0
        for d in range(1, n + 1):
            for r in range(d):
                cur_sum = 0
                max_pos = 0
                min_neg = 0
                for i in range(r, n, d):
                    cur_sum += a[i]
                    if cur_sum > max_pos:
                        max_pos = cur_sum
                    if cur_sum < min_neg:
                        min_neg = cur_sum
                if max_pos > max_abs:
                    max_abs = max_pos
                if -min_neg > max_abs:
                    max_abs = -min_neg
        return max_abs

    # Initialize with Thue-Morse
    signs = []
    for i in range(n):
        c = bin(i).count('1')
        signs.append(1 if c % 2 == 0 else -1)
    
    current_signs = signs[:]
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 55.0
    
    # Check if Thue-Morse is already optimal for K=2
    if find_violation(current_signs, 2) is None:
        return current_signs

    # Guided Local Search to reach K=2
    best_signs = current_signs[:]
    best_disc = calc_max_disc(best_signs)
    
    while time.time() - start_time < time_limit - 5.0: # Leave 5s for final SA if needed
        viol = find_violation(current_signs, 2)
        
        if viol is None:
            # Found a solution with discrepancy <= 2
            # Verify it
            if calc_max_disc(current_signs) <= 2:
                return current_signs
            # If calc says > 2, there's a bug in find_violation or logic, 
            # but find_violation is exact for > K. 
            # So if find_violation returns None, disc <= 2.
            return current_signs
        
        d, r, peak_i = viol
        
        # Determine violation sign
        cur_sum = 0
        for i in range(r, peak_i + 1, d):
            cur_sum += current_signs[i]
            
        indices_in_ap = list(range(r, peak_i + 1, d))
        
        if cur_sum > 0:
            candidates = [i for i in indices_in_ap if current_signs[i] == 1]
        else:
            candidates = [i for i in indices_in_ap if current_signs[i] == -1]
            
        if not candidates:
            candidates = indices_in_ap
            
        idx = rng.choice(candidates)
        current_signs[idx] = -current_signs[idx]
        
        # Track best
        # Only calculate full disc occasionally to save time
        if rng.random() < 0.01:
            disc = calc_max_disc(current_signs)
            if disc < best_disc:
                best_disc = disc
                best_signs = current_signs[:]
                
        # Random escape
        if rng.random() < 0.05:
            idx = rng.randint(0, n - 1)
            current_signs[idx] = -current_signs[idx]

    # If we didn't find K=2, try to minimize further with SA on the best found
    # Use remaining time
    T = 1.0
    while time.time() - start_time < time_limit:
        idx = rng.randint(0, n - 1)
        old_val = current_signs[idx]
        current_signs[idx] = -old_val
        
        # Fast check: is it better than best?
        # We can't easily know without full calc.
        # Just do full calc if we are close to time limit or randomly.
        # Actually, let's just do full calc for every step in this short phase.
        new_disc = calc_max_disc(current_signs)
        
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = current_signs[:]
        else:
            # Metropolis
            if T > 0:
                prob = math.exp(-(new_disc - best_disc) / T)
                if rng.random() < prob:
                    pass # Keep
                else:
                    current_signs[idx] = old_val
            else:
                current_signs[idx] = old_val
        
        T *= 0.99
        if T < 0.01:
            T = 0.01

    return best_signs

# EVOLVE-BLOCK-END
