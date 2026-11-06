import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    # Precompute weights w_i = L // i (1-indexed: weights[1]..weights[30])
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = L // i

    # Helper: compute sum from coeffs (0-indexed)
    def compute_sum(coeffs):
        s = 0
        for i in range(N):
            if coeffs[i] != 0:
                s += coeffs[i] * weights[i + 1]
        return s

    # =====================
    # PHASE 1: Simulated Annealing
    # =====================
    # Initialize with random coefficients
    coeffs = [random.choice([-1, 0, 1]) for _ in range(N)]
    current_sum = compute_sum(coeffs)
    
    best_abs_num = float('inf')
    best_coeffs = [1] + [0] * (N - 1)  # Fallback
    
    if current_sum != 0:
        best_abs_num = abs(current_sum)
        best_coeffs = coeffs[:]
        if best_abs_num == 1:
            return best_coeffs

    # SA parameters
    T_start = L
    T_end = 1
    # Estimate ~3M steps in 60s
    total_steps_estimate = 3000000
    alpha = (T_end / T_start) ** (1.0 / total_steps_estimate)
    
    T = T_start
    step = 0
    
    # Run SA until time limit
    while time.time() < deadline - 20.0:  # Leave 20s for phase 2
        # Pick random index
        i = random.randint(0, N - 1)
        old_val = coeffs[i]
        
        # Pick new value from {-1, 0, 1} \ {old_val}
        if old_val == -1:
            new_val = random.choice([0, 1])
        elif old_val == 0:
            new_val = random.choice([-1, 1])
        else:
            new_val = random.choice([-1, 0])
            
        delta = (new_val - old_val) * weights[i + 1]
        new_sum = current_sum + delta
        
        old_cost = abs(current_sum) if current_sum != 0 else L
        new_cost = abs(new_sum) if new_sum != 0 else L
        
        # Acceptance criterion
        if new_cost < old_cost:
            coeffs[i] = new_val
            current_sum = new_sum
            step += 1
            T *= alpha
            
            if new_sum != 0 and new_cost < best_abs_num:
                best_abs_num = new_cost
                best_coeffs = coeffs[:]
                if best_abs_num == 1:
                    return best_coeffs
        else:
            if new_cost == old_cost:
                if random.random() < 0.5:
                    coeffs[i] = new_val
                    current_sum = new_sum
                    step += 1
                    T *= alpha
            else:
                delta_E = new_cost - old_cost
                if delta_E > T * 20:
                    pass
                else:
                    prob = 2.718281828 ** (-delta_E / T)
                    if random.random() < prob:
                        coeffs[i] = new_val
                        current_sum = new_sum
                        step += 1
                        T *= alpha
        
        if time.time() > deadline - 20.5:
            break

    # =====================
    # PHASE 2: Local Search Refinement
    # =====================
    # Use best_coeffs as starting point for focused local search
    if time.time() < deadline - 5.0:
        ls_coeffs = best_coeffs[:]
        ls_sum = compute_sum(ls_coeffs)
        
        # Hill climbing with multiple restarts
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # If restart > 0, perturb the solution
            if restart > 0:
                for _ in range(5):
                    i = random.randint(0, N - 1)
                    old_val = ls_coeffs[i]
                    if old_val == -1:
                        ls_coeffs[i] = random.choice([0, 1])
                    elif old_val == 0:
                        ls_coeffs[i] = random.choice([-1, 1])
                    else:
                        ls_coeffs[i] = random.choice([-1, 0])
                ls_sum = compute_sum(ls_coeffs)
            
            # Hill climb
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5.0:
                        break
                    old_val = ls_coeffs[i]
                    for new_val in [-1, 0, 1]:
                        if new_val == old_val:
                            continue
                        delta = (new_val - old_val) * weights[i + 1]
                        new_sum = ls_sum + delta
                        new_cost = abs(new_sum) if new_sum != 0 else L
                        old_cost = abs(ls_sum) if ls_sum != 0 else L
                        
                        if new_cost < old_cost:
                            ls_coeffs[i] = new_val
                            ls_sum = new_sum
                            improved = True
                            
                            if new_sum != 0 and new_cost < best_abs_num:
                                best_abs_num = new_cost
                                best_coeffs = ls_coeffs[:]
                                if best_abs_num == 1:
                                    return best_coeffs
                            break  # Only one change per iteration

    # =====================
    # PHASE 3: Sparse Enumeration (k=2 to 5)
    # =====================
    # Quick pass over small support sets to catch sparse optima
    if time.time() < deadline - 1.0:
        import itertools
        
        max_k = 5  # Keep it fast
        for k in range(2, max_k + 1):
            if time.time() > deadline - 1.0:
                break
            
            indices = range(1, N + 1)
            for combo in itertools.combinations(indices, k):
                if time.time() > deadline - 1.0:
                    break
                
                total_pos = 0
                for idx in combo:
                    total_pos += weights[idx]
                
                for mask in range(1 << k):
                    subset_sum = 0
                    for j in range(k):
                        if (mask >> j) & 1:
                            subset_sum += weights[combo[j]]
                    
                    num = total_pos - 2 * subset_sum
                    abs_num = abs(num)
                    
                    if abs_num > 0 and abs_num < best_abs_num:
                        best_abs_num = abs_num
                        coeffs = [0] * N
                        for j in range(k):
                            idx = combo[j]
                            if (mask >> j) & 1:
                                coeffs[idx - 1] = -1
                            else:
                                coeffs[idx - 1] = 1
                        best_coeffs = coeffs
                        
                        if best_abs_num == 1:
                            return best_coeffs

    # Final validation
    final_sum = compute_sum(best_coeffs)
    if final_sum == 0:
        # Force non-zero
        for i in range(N):
            if best_coeffs[i] == 0:
                best_coeffs[i] = 1
                break
            elif best_coeffs[i] == 1:
                best_coeffs[i] = -1
                break
            else:
                best_coeffs[i] = 0
                break
        final_sum = compute_sum(best_coeffs)
        if final_sum == 0:
            best_coeffs[0] = 1
            best_coeffs[1] = 0
            best_coeffs[2] = 0

    return best_coeffs
