import time
import random
from math import gcd
from functools import reduce
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i
        
    def get_sum(c):
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * A[i + 1]
        return s

    # Initial solutions
    c1 = [0] * 28 + [1, -1]
    val1 = get_sum(c1)
    
    c2 = [1 if i % 2 == 0 else -1 for i in range(30)]
    val2 = get_sum(c2)
    
    random.seed(42)
    c3 = [random.choice([-1, 0, 1]) for _ in range(30)]
    val3 = get_sum(c3)
    if val3 == 0:
        c3[0] = 1
        val3 = get_sum(c3)
        
    candidates = [(abs(val1), c1, val1), (abs(val2), c2, val2), (abs(val3), c3, val3)]
    candidates.sort(key=lambda x: x[0])
    
    best_abs = candidates[0][0]
    best_val = candidates[0][2]
    best_c = candidates[0][1][:]
    
    vals = (-1, 0, 1)
    
    def simulated_annealing(initial_c, initial_val, max_time):
        nonlocal best_abs, best_val, best_c
        start_sa = time.time()
        
        c = initial_c[:]
        current_val = initial_val
        current_abs = abs(current_val)
        
        # Temperature
        T = float(L) / 50.0
        T_min = 1.0
        cooling = 0.9999
        
        while time.time() - start_sa < max_time:
            i = random.randint(0, 29)
            old_val = c[i]
            
            new_val = vals[random.randint(0, 2)]
            if new_val == old_val:
                continue
                
            delta = (new_val - old_val) * A[i + 1]
            new_sum = current_val + delta
            new_abs = abs(new_sum)
            
            if new_abs < current_abs:
                c[i] = new_val
                current_val = new_sum
                current_abs = new_abs
                
                if new_abs < best_abs and new_sum != 0:
                    best_abs = new_abs
                    best_val = new_sum
                    best_c = c[:]
            else:
                if new_sum == 0:
                    continue
                
                delta_e = new_abs - current_abs
                # Acceptance probability
                # If delta_e is large, exp is small
                # We can skip if delta_e > T * 20 (approx 2e-9 prob)
                if delta_e < T * 20:
                    try:
                        prob = math.exp(-delta_e / T)
                    except OverflowError:
                        prob = 0.0
                    if random.random() < prob:
                        c[i] = new_val
                        current_val = new_sum
                        current_abs = new_abs
                        if new_abs < best_abs:
                            best_abs = new_abs
                            best_val = new_sum
                            best_c = c[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_abs

    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 1.0:
        # Run SA for the remaining time
        # Split into 2 runs to allow restarts
        time_per_run = remaining_time / 2.0
        
        # Run 1 from best candidate
        simulated_annealing(candidates[0][1], candidates[0][2], time_per_run)
        
        # Run 2 from second best
        if time.time() - start_time < deadline:
            simulated_annealing(candidates[1][1], candidates[1][2], deadline - (time.time() - start_time))
            
    return best_c
