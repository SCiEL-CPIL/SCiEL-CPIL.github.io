"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def construct():
    deadline = time.time() + 85
    
    best_coeffs = [0] * 30
    best_abs_val = float('inf')
    best_num = 0
    best_den = 1
    
    # Precompute weights 1/i
    weights = [1.0 / (i + 1) for i in range(30)]
    
    def random_coeffs():
        return [random.choice([-1, 0, 1]) for _ in range(30)]
    
    def get_exact(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += Fraction(c, i + 1)
        return s
        
    def update_best(coeffs, val):
        nonlocal best_coeffs, best_abs_val, best_num, best_den
        if val < best_abs_val:
            best_abs_val = val
            best_coeffs = list(coeffs)
            s = get_exact(coeffs)
            best_num = abs(s.numerator)
            best_den = s.denominator
            # If we found 1/L, we can stop early if we want, but let's keep searching for more
            # Actually, 1 is the minimum possible non-zero numerator.
            if best_num == 1:
                return True
        return False

    def optimize(initial_coeffs, time_limit):
        nonlocal best_abs_val, best_num
        coeffs = list(initial_coeffs)
        cur_sum = 0.0
        for i, c in enumerate(coeffs):
            cur_sum += c * weights[i]
            
        if cur_sum == 0:
            # Fix zero
            idx = random.randint(0, 29)
            coeffs[idx] = random.choice([-1, 1])
            cur_sum += coeffs[idx] * weights[idx]
            
        # Initial best check
        if abs(cur_sum) < best_abs_val:
            if update_best(coeffs, abs(cur_sum)):
                return True
                
        T = 1.0
        T_min = 1e-12
        cooling = 0.99995
        
        start_time = time.time()
        
        while time.time() < start_time + time_limit:
            if best_num == 1:
                return True
                
            i = random.randint(0, 29)
            old_val = coeffs[i]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
                
            delta = (new_val - old_val) * weights[i]
            new_sum = cur_sum + delta
            
            if new_sum == 0:
                continue
                
            new_abs = abs(new_sum)
            cur_abs = abs(cur_sum)
            
            if new_abs < cur_abs:
                coeffs[i] = new_val
                cur_sum = new_sum
                cur_abs = new_abs
                
                # Check for improvement in global best
                if cur_abs < best_abs_val:
                    if update_best(coeffs, cur_abs):
                        return True
                        
            else:
                diff = new_abs - cur_abs
                if T > T_min:
                    # Optimization: if diff is large, exp is 0
                    if diff / T < 50: 
                        prob = math.exp(-diff / T)
                    else:
                        prob = 0.0
                    if random.random() < prob:
                        coeffs[i] = new_val
                        cur_sum = new_sum
                        cur_abs = new_abs
                        
            T *= cooling
            if T < T_min:
                T = T_min

        return False

    random.seed(42)
    
    # Phase 1: Quick random starts
    for _ in range(20):
        if time.time() > deadline:
            break
        c = random_coeffs()
        if optimize(c, 1.0):
            break
            
    # Phase 2: Long runs
    if time.time() < deadline - 5:
        c = random_coeffs()
        remaining = deadline - time.time() - 2
        if optimize(c, remaining):
            pass

    # Phase 3: If best is still poor, try specific patterns
    if best_abs_val > 0.1:
        # Alternating
        c_alt = [1 if i % 2 == 0 else -1 for i in range(30)]
        s_alt = sum(c * w for c, w in zip(c_alt, weights))
        if s_alt != 0 and abs(s_alt) < best_abs_val:
            update_best(c_alt, abs(s_alt))
            
        # Random restarts with local search
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            c = random_coeffs()
            optimize(c, 0.2)

    return best_coeffs

# EVOLVE-BLOCK-END
