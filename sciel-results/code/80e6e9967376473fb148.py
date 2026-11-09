import time
import random
import math
from functools import reduce
from math import gcd

def construct():
    start_time = time.time()
    deadline = 58.0  # safety margin

    # Precompute LCM of 1..30 for exact rational arithmetic
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i

    # Initial solution: [1, -1, 0, ...] gives sum 1 - 1/2 = 1/2
    # Integer sum: L - L/2 = L/2
    best_abs = L // 2
    best_c = [0] * 30
    best_c[0] = 1
    best_c[1] = -1

    def get_sum(c):
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * A[i + 1]
        return s

    def run_sa(iterations, t0, t1):
        nonlocal best_abs, best_c
        
        # Bind to locals for speed
        A_loc = A
        rand_int = random.randrange
        rand_choice = random.choice
        rand_float = random.random
        exp = math.exp
        
        # Random initial state
        c = [rand_choice([-1, 0, 1]) for _ in range(30)]
        current_sum = 0
        for i in range(30):
            if c[i] != 0:
                current_sum += c[i] * A_loc[i + 1]
        
        # If sum is 0, it's invalid. Nudge it.
        if current_sum == 0:
            # Flip a random bit
            idx = rand_int(30)
            if c[idx] == 0:
                c[idx] = 1
            elif c[idx] == 1:
                c[idx] = -1
            else:
                c[idx] = 1
            # Recalculate
            current_sum = 0
            for i in range(30):
                if c[i] != 0:
                    current_sum += c[i] * A_loc[i + 1]
            if current_sum == 0:
                c[0] = 1
                current_sum += A_loc[1]

        current_abs = abs(current_sum)
        
        # Update best if improved
        if current_abs < best_abs:
            best_abs = current_abs
            best_c = c[:]

        # Annealing parameters
        if iterations <= 0:
            return
            
        alpha = (t1 / t0) ** (1.0 / iterations)
        T = t0
        
        for k in range(iterations):
            if time.time() - start_time > deadline:
                return

            # Pick random index
            i = rand_int(30)
            old_val = c[i]
            
            # Pick new value different from old
            if old_val == 0:
                new_val = rand_choice([-1, 1])
            elif old_val == 1:
                new_val = rand_choice([-1, 0])
            else:
                new_val = rand_choice([1, 0])
            
            c[i] = new_val
            delta = (new_val - old_val) * A_loc[i + 1]
            new_sum = current_sum + delta
            
            if new_sum == 0:
                # Invalid state, revert
                c[i] = old_val
                T *= alpha
                continue
            
            new_abs = abs(new_sum)
            
            # Acceptance criterion
            if new_abs < current_abs:
                # Always accept improvement
                current_sum = new_sum
                current_abs = new_abs
                if current_abs < best_abs:
                    best_abs = current_abs
                    best_c = c[:]
            else:
                # Deterioration
                delta_abs = new_abs - current_abs
                
                # Accept with probability exp(-delta_abs / T)
                # Optimization: if T is very small, reject
                if T < 1e-9:
                    accept = False
                else:
                    # If delta_abs is very large compared to T, prob is ~0
                    if delta_abs > T * 20:
                        accept = False
                    else:
                        p = exp(-delta_abs / T)
                        accept = rand_float() < p
            
            if not accept:
                # Revert
                c[i] = old_val
            
            T *= alpha

    # Run SA with multiple restarts
    T0 = L
    T1 = 1.0
    
    iterations_per_run = 50000
    
    while time.time() - start_time < deadline:
        run_sa(iterations_per_run, T0, T1)
        
        # If we found the theoretical minimum (1), stop.
        if best_abs == 1:
            break
            
    return best_c
