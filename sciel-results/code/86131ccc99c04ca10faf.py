import time
import random

def construct():
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for 150s total
    
    N = 512
    K = 32
    
    def set_to_bits(s):
        bits = 0
        for x in s:
            bits |= (1 << (x - 1))
        return bits
            
    def bits_to_set(bits):
        return [i + 1 for i in range(N) if (bits >> i) & 1]

    def score_bits(bits):
        count = 0
        for d in range(1, 256):
            if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
                count += 1
        return count

    # Precompute: for each element x, which d values would be "covered" if x is in the set
    # Actually, let's precompute a score mask approach:
    # For each x in 1..N, precompute which d's are "activated" by x being present
    # This is complex, so let's use a different approach:
    # Precompute for each pair (a, b) with a < b, the d = b - a
    # Then for each x, we can track which (a, b) pairs are completed by x
    
    # Alternative: use numpy for faster bit operations if available
    try:
        import numpy as np
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False
    
    # 1. Greedy Initialization
    S = set()
    current_bits = 0
    
    for _ in range(K):
        best_x = -1
        best_score = -1
        for x in range(1, N + 1):
            if x in S:
                continue
            new_bits = current_bits | (1 << (x - 1))
            sc = score_bits(new_bits)
            if sc > best_score:
                best_score = sc
                best_x = x
        
        if best_x != -1:
            S.add(best_x)
            current_bits |= (1 << (best_x - 1))
            
    current_score = best_score
    best_score = current_score
    best_bits = current_bits
    current_set = S
    
    # 2. Simulated Annealing Polish
    if time.time() - start_time > time_limit - 10:
        return bits_to_set(best_bits)
        
    T_start = 3.0
    T_end = 0.001
    
    # Track stagnation for restarts
    stagnation_count = 0
    max_stagnation = 2000
    best_so_far_score = best_score
    
    while time.time() - start_time < time_limit - 5:
        # Pick random x in S, y not in S
        x = random.choice(list(current_set))
        y = random.randint(1, N)
        
        if y in current_set:
            continue
            
        # Compute new bits
        new_bits = current_bits
        new_bits &= ~(1 << (x - 1))
        new_bits |= (1 << (y - 1))
        
        new_score = score_bits(new_bits)
        delta = new_score - current_score
        
        # Calculate T based on time elapsed
        t_elapsed = time.time() - start_time
        progress = t_elapsed / time_limit
        # Clamp progress to [0, 1]
        if progress > 1.0:
            progress = 1.0
        T = T_start * pow(T_end / T_start, progress)
        
        if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
            # Accept
            current_bits = new_bits
            current_set.remove(x)
            current_set.add(y)
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits
                best_so_far_score = best_score
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1
        
        # If stuck for too long, do a random restart with perturbation
        if stagnation_count > max_stagnation:
            # Perturb the current best solution
            perturbed_bits = best_bits
            perturbed_set = set(bits_to_set(best_bits))
            
            # Remove 2-3 random elements and add new ones
            num_remove = random.randint(2, 4)
            for _ in range(num_remove):
                if perturbed_set:
                    rem = random.choice(list(perturbed_set))
                    perturbed_set.remove(rem)
                    perturbed_bits &= ~(1 << (rem - 1))
            
            # Add new random elements
            for _ in range(num_remove):
                new_elem = random.randint(1, N)
                if new_elem not in perturbed_set:
                    perturbed_set.add(new_elem)
                    perturbed_bits |= (1 << (new_elem - 1))
            
            current_bits = perturbed_bits
            current_set = perturbed_set
            current_score = score_bits(current_bits)
            stagnation_count = 0
    
    # 3. Final polish: pure hill climbing on best solution
    if time.time() - start_time < time_limit - 2:
        polish_bits = best_bits
        polish_set = set(bits_to_set(best_bits))
        polish_score = score_bits(polish_bits)
        
        improved = True
        while improved and time.time() - start_time < time_limit - 2:
            improved = False
            for x in list(polish_set):
                for y in range(1, N + 1):
                    if y in polish_set:
                        continue
                    new_bits = polish_bits
                    new_bits &= ~(1 << (x - 1))
                    new_bits |= (1 << (y - 1))
                    new_score = score_bits(new_bits)
                    if new_score > polish_score:
                        polish_bits = new_bits
                        polish_set.remove(x)
                        polish_set.add(y)
                        polish_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if polish_score > best_score:
            best_score = polish_score
            best_bits = polish_bits
    
    return bits_to_set(best_bits)
