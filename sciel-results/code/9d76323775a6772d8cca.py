# EVOLVE-BLOCK-START
from fractions import Fraction
import random
import math
import time

def construct():
    random.seed(42)
    
    N = 30
    recips = [Fraction(1, i + 1) for i in range(N)]
    
    def calc_sum(coeff):
        s = Fraction(0, 1)
        for i, c in enumerate(coeff):
            if c != 0:
                s += c * recips[i]
        return s

    best_global_coeff = None
    best_global_abs = None
    best_global_sum = None
    
    start_time = time.time()
    max_time = 58.0 # Leave 2s buffer for safety
    
    # Parameters for SA
    # We will run multiple independent chains
    # Each chain: 
    #   Init: Balanced random
    #   Steps: ~50,000
    #   T_start: 1.0, T_end: 1e-8
    
    while time.time() - start_time < max_time:
        # 1. Initialize
        coeff = [0] * N
        # Ensure some non-zeros to avoid trivial zero sum start
        num_pos = random.randint(5, 15)
        num_neg = num_pos
        indices = random.sample(range(N), num_pos + num_neg)
        for k in range(num_pos):
            coeff[indices[k]] = 1
        for k in range(num_neg):
            coeff[indices[num_pos + k]] = -1
        
        current_coeff = coeff[:]
        current_sum = calc_sum(current_coeff)
        
        if current_sum == 0:
            # Perturb
            coeff[0] = 1
            current_coeff = coeff[:]
            current_sum = calc_sum(current_coeff)
            if current_sum == 0:
                coeff[1] = 1
                current_coeff = coeff[:]
                current_sum = calc_sum(current_coeff)
        
        best_coeff = current_coeff[:]
        best_sum = current_sum
        best_abs = abs(current_sum)
        current_E = abs(float(current_sum))
        
        # SA Parameters
        steps = 50000
        T_start = 1.0
        T_end = 1e-8
        decay = (T_end / T_start) ** (1.0 / steps)
        T_cur = T_start
        
        for _ in range(steps):
            # Select index
            i = random.randint(0, N - 1)
            old_val = current_coeff[i]
            
            # Select new value
            if old_val == 0:
                new_val = 1 if random.random() < 0.5 else -1
            elif old_val == 1:
                new_val = 0 if random.random() < 0.5 else -1
            else: # -1
                new_val = 0 if random.random() < 0.5 else 1
                
            # Update sum
            delta = (new_val - old_val) * recips[i]
            new_sum = current_sum + delta
            
            # Check zero
            if new_sum == 0:
                T_cur *= decay
                continue
            
            # Energy
            new_E = abs(float(new_sum))
            delta_E = new_E - current_E
            
            # Acceptance
            if delta_E < 0:
                accept = True
            else:
                if T_cur < 1e-9:
                    accept = False
                else:
                    # P = exp(-delta_E / T_cur)
                    # Optimization: if delta_E is very large, exp is 0.
                    # if delta_E / T_cur > 700, exp is 0.
                    ratio = delta_E / T_cur
                    if ratio > 700:
                        accept = False
                    else:
                        accept = random.random() < math.exp(-ratio)
            
            if accept:
                current_coeff[i] = new_val
                current_sum = new_sum
                current_E = new_E
                
                abs_new = abs(new_sum)
                if abs_new < best_abs:
                    best_abs = abs_new
                    best_sum = new_sum
                    best_coeff = current_coeff[:]
            
            T_cur *= decay
        
        # End of run
        if best_global_abs is None or best_abs < best_global_abs:
            best_global_abs = best_abs
            best_global_sum = best_sum
            best_global_coeff = best_coeff

    if best_global_coeff is None:
        best_global_coeff = [0] * 30
        best_global_coeff[0] = 1
        
    return best_global_coeff
# EVOLVE-BLOCK-END
