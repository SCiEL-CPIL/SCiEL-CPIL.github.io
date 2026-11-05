import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = time.time() + 58
    
    n = 30
    L = 2329089562800
    w = [L // i for i in range(1, n + 1)]
    
    best_abs = float('inf')
    best_coeffs = None
    
    def update_best(coeffs, current_sum):
        nonlocal best_abs, best_coeffs
        if current_sum != 0:
            abs_sum = abs(current_sum)
            if abs_sum < best_abs:
                best_abs = abs_sum
                best_coeffs = list(coeffs)
                
    # Simulated Annealing Parameters
    T_start = 100000.0  # High enough to explore
    T_end = 1.0
    alpha = 0.99
    
    # Multi-start
    num_restarts = 0
    while time.time() < deadline - 2:
        num_restarts += 1
        # Random init
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        current_sum = sum(coeffs[i] * w[i] for i in range(n))
        
        if current_sum != 0 and abs(current_sum) < best_abs:
            best_abs = abs(current_sum)
            best_coeffs = list(coeffs)
            
        T = T_start
        stuck_counter = 0
        
        # Inner loop
        while time.time() < deadline - 2:
            # Choose move type: 1-bit or 2-bit
            if random.random() < 0.7:
                # 1-bit move
                i = random.randint(0, n - 1)
                old_val = coeffs[i]
                options = [v for v in [-1, 0, 1] if v != old_val]
                new_val = random.choice(options)
                
                delta = (new_val - old_val) * w[i]
                new_sum = current_sum + delta
                new_abs = abs(new_sum)
                
                old_abs = abs(current_sum)
                
                if new_abs < old_abs:
                    coeffs[i] = new_val
                    current_sum = new_sum
                    stuck_counter = 0
                    update_best(coeffs, current_sum)
                else:
                    delta_E = new_abs - old_abs
                    if delta_E == 0:
                        coeffs[i] = new_val
                        current_sum = new_sum
                        # update_best not needed if abs same, but check just in case?
                        # No, abs is same.
                    else:
                        # Accept with prob
                        # Optimization: if delta_E is large, prob is 0
                        # w[i] is at least L/30 ~ 7e10.
                        # If T is small, prob is 0.
                        if delta_E < T * 50: # Heuristic cutoff
                            try:
                                prob = math.exp(-delta_E / T)
                            except OverflowError:
                                prob = 0.0
                            if random.random() < prob:
                                coeffs[i] = new_val
                                current_sum = new_sum
                                stuck_counter = 0
                                update_best(coeffs, current_sum)
                # If we are at 0, we might get stuck.
                if current_sum == 0:
                    stuck_counter += 1
                    if stuck_counter > 100:
                        # Force a move away from 0
                        # Pick a random i and set to non-zero
                        i = random.randint(0, n - 1)
                        if coeffs[i] == 0:
                            coeffs[i] = random.choice([-1, 1])
                        else:
                            coeffs[i] = random.choice([-1, 1])
                            if coeffs[i] == coeffs[i]: # This check is bad
                                coeffs[i] = -coeffs[i] if coeffs[i] != 0 else 1
                        current_sum = sum(coeffs[k] * w[k] for k in range(n))
                        stuck_counter = 0
            else:
                # 2-bit move
                i, j = random.sample(range(n), 2)
                vi = coeffs[i]
                vj = coeffs[j]
                
                # Random new values
                vi_new = random.choice([v for v in [-1, 0, 1] if v != vi])
                vj_new = random.choice([v for v in [-1, 0, 1] if v != vj])
                
                delta = (vi_new - vi) * w[i] + (vj_new - vj) * w[j]
                new_sum = current_sum + delta
                new_abs = abs(new_sum)
                old_abs = abs(current_sum)
                
                if new_abs < old_abs:
                    coeffs[i] = vi_new
                    coeffs[j] = vj_new
                    current_sum = new_sum
                    stuck_counter = 0
                    update_best(coeffs, current_sum)
                else:
                    delta_E = new_abs - old_abs
                    if delta_E == 0:
                        coeffs[i] = vi_new
                        coeffs[j] = vj_new
                        current_sum = new_sum
                    else:
                        if delta_E < T * 50:
                            try:
                                prob = math.exp(-delta_E / T)
                            except OverflowError:
                                prob = 0.0
                            if random.random() < prob:
                                coeffs[i] = vi_new
                                coeffs[j] = vj_new
                                current_sum = new_sum
                                stuck_counter = 0
                                update_best(coeffs, current_sum)
            
            T *= alpha
            if T < T_end:
                T = T_end

    if best_coeffs is None:
        return [1, -1] + [0] * 28
        
    return best_coeffs

# EVOLVE-BLOCK-END
