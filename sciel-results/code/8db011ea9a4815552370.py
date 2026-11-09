import time
import random
from math import gcd
from functools import reduce
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    # Precompute LCM of 1..30 for exact rational arithmetic
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i

    # Initial solution: random non-zero
    random.seed(42)
    c = [random.choice([-1, 0, 1]) for _ in range(30)]
    
    def get_sum(c):
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * A[i + 1]
        return s
        
    current_sum = get_sum(c)
    if current_sum == 0:
        # Ensure non-zero
        c[0] = 1
        current_sum = get_sum(c)
        if current_sum == 0:
             c[1] = 1
             current_sum = get_sum(c)
             
    best_c = c[:]
    best_sum = current_sum
    best_abs = abs(best_sum)
    
    # Simulated Annealing parameters
    # Max possible sum magnitude is approx 4 * L
    # L is approx 2.3e12
    # So max delta is approx 1e13
    # Increase T to allow larger jumps to escape local minima
    T = 2.0e12 
    T_min = 1.0
    
    # Estimate steps: 55s / ~50ns per step (optimistic) ~ 1e9 steps? 
    # Realistically with Python overhead, ~1e7-1e8 steps.
    # Let's target 1e8 steps.
    # alpha = (T_min / T)^(1/N)
    # ln(alpha) = (ln(T_min) - ln(T)) / N
    # ln(1) - ln(2e12) = -28.3
    # If N = 1e8, ln(alpha) = -28.3e-8 = -2.83e-7
    # alpha = exp(-2.83e-7) approx 1 - 2.83e-7
    alpha = 0.999999717
    
    # Local variables for speed
    rand_int = random.randint
    rand_choice = random.choice
    rand_float = random.random
    exp_func = math.exp
    
    while time.time() - start_time < deadline and T > T_min:
        # Pick a random index
        i = rand_int(0, 29)
        old_val = c[i]
        # Pick a new value
        new_val = rand_choice([-1, 0, 1])
        if new_val == old_val:
            continue
            
        c[i] = new_val
        new_sum = current_sum + (new_val - old_val) * A[i + 1]
        
        if new_sum == 0:
            # Reject zero sum
            c[i] = old_val
            T *= alpha
            continue
            
        new_abs = abs(new_sum)
        current_abs = abs(current_sum)
        
        delta = new_abs - current_abs
        
        if delta < 0:
            # Accept improvement
            current_sum = new_sum
            if new_abs < best_abs:
                best_abs = new_abs
                best_sum = new_sum
                best_c = c[:]
        else:
            # Metropolis criterion
            # Prob = exp(-delta / T)
            if delta / T < 700: # Avoid overflow/underflow issues, exp(-700) is ~0
                prob = exp_func(-delta / T)
            else:
                prob = 0.0
                
            if rand_float() < prob:
                current_sum = new_sum
            else:
                # Revert
                c[i] = old_val
        
        T *= alpha
        
    # Final polish: Greedy local search for a short time
    # Try to improve best_c by single flips
    polish_deadline = 58.0
    while time.time() - start_time < polish_deadline:
        improved = False
        for i in range(30):
            old_val = best_c[i]
            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                best_c[i] = new_val
                new_sum = best_sum + (new_val - old_val) * A[i + 1]
                if new_sum != 0:
                    new_abs = abs(new_sum)
                    if new_abs < best_abs:
                        best_abs = new_abs
                        best_sum = new_sum
                        improved = True
                        break
                best_c[i] = old_val
            if improved:
                break
        if not improved:
            break

    return best_c
