import numpy as np
from math import gcd
from functools import reduce
import time
import random

def construct():
    n = 30
    # LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n+1))
    
    # Scaled values
    vals = [L // i for i in range(1, n+1)]
    
    # Simulated Annealing
    # State: list of coefficients
    # Cost: absolute sum (0 is invalid, treat as high cost)
    
    best_sum = None
    best_coeffs = None
    best_cost = float('inf')
    
    # Time limit
    start_time = time.time()
    time_limit = 50.0 # Leave 10s buffer
    
    # Initial random state
    # To be fast, we can try a few different initializations
    # or just one good one.
    
    # Helper to calculate sum
    def calc_sum(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * vals[i]
        return s
        
    # Helper to get cost
    def get_cost(s):
        if s == 0:
            return float('inf')
        return abs(s)

    # Run SA
    # We will run multiple iterations/restarts if time permits
    
    num_restarts = 5
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize coeffs
        # Random
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        current_sum = calc_sum(coeffs)
        current_cost = get_cost(current_sum)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_sum = current_sum
            best_coeffs = coeffs[:]
            
        # SA Parameters
        T = 1000000.0 # High initial temperature
        T_min = 1.0
        cooling_rate = 0.9999
        
        # Max iterations per restart
        # Estimate: 100k iterations per second?
        # 50s / 5 restarts = 10s per restart.
        # Let's just loop until time is up for this restart?
        # Or fixed iterations.
        
        while time.time() - start_time < time_limit:
            # Pick random index
            i = random.randint(0, n - 1)
            # Pick new value
            old_val = coeffs[i]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
                
            # Calculate delta
            # delta = |new_sum| - |old_sum|
            # But we have the 0 constraint.
            
            old_abs = abs(current_sum) if current_sum != 0 else float('inf')
            
            coeffs[i] = new_val
            new_sum = current_sum + (new_val - old_val) * vals[i]
            new_abs = abs(new_sum) if new_sum != 0 else float('inf')
            
            # Check if new is better
            if new_abs < old_abs:
                current_sum = new_sum
                current_cost = new_abs
                if new_abs < best_cost:
                    best_cost = new_abs
                    best_sum = new_sum
                    best_coeffs = coeffs[:]
            else:
                # Accept with probability exp(-delta / T)
                delta = new_abs - old_abs
                if delta == 0:
                    current_sum = new_sum
                else:
                    # If old was inf (0), delta is inf?
                    # If old_sum == 0, old_abs = inf.
                    # If new_sum != 0, new_abs is finite.
                    # delta = finite - inf = -inf.
                    # exp(-(-inf)/T) = exp(inf) = inf?
                    # We should always accept if moving from 0 to non-zero.
                    if old_abs == float('inf'):
                        current_sum = new_sum
                    else:
                        # If new is inf (0), delta = inf.
                        # exp(-inf/T) = 0.
                        # So we reject moving to 0.
                        if new_abs == float('inf'):
                            pass # Reject
                        else:
                            prob = np.exp(-delta / T)
                            if random.random() < prob:
                                current_sum = new_sum
                                current_cost = new_abs
                                if new_abs < best_cost:
                                    best_cost = new_abs
                                    best_sum = new_sum
                                    best_coeffs = coeffs[:]
            
            T *= cooling_rate
            if T < T_min:
                T = T_min

    if best_coeffs is None:
        # Fallback
        return [0] * 28 + [1, -1]
        
    return best_coeffs
