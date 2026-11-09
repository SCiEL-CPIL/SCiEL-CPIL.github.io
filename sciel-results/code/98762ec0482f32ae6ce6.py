import time
import random
import math
from math import gcd, exp

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 30
    
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    best_sum = None
    best_coeffs = None
    
    def evaluate(coeffs):
        s = 0
        for i in range(n):
            if coeffs[i] != 0:
                s += coeffs[i] * weights[i]
        return s
    
    def local_search(coeffs, current_sum):
        """Hill climbing to find local minimum"""
        improved = True
        max_iters = 200
        iter_count = 0
        while improved and iter_count < max_iters:
            iter_count += 1
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                old_val = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[i]
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    if abs(new_sum) < abs(current_sum):
                        coeffs[i] = new_val
                        current_sum = new_sum
                        improved = True
                        break
        return coeffs, current_sum
    
    def simulated_annealing():
        nonlocal best_sum, best_coeffs
        
        # Random initial configuration
        current = [random.choice([-1, 0, 1]) for _ in range(n)]
        current_sum = evaluate(current)
        
        # If we start at 0, make a small perturbation
        if current_sum == 0:
            idx = random.randint(0, n - 1)
            current[idx] = random.choice([-1, 1])
            current_sum = evaluate(current)
        
        # Track best non-zero solution found during this run
        run_best_sum = current_sum if current_sum != 0 else None
        run_best_coeffs = current[:] if current_sum != 0 else None
        
        # Annealing parameters - reduced iterations for speed
        T_start = 500000000.0
        T_end = 1.0
        iterations = 10000
        cooling_rate = (T_end / T_start) ** (1.0 / iterations)
        
        T = T_start
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Pick a random index to modify
            i = random.randint(0, n - 1)
            old_val = current[i]
            
            # Choose a new value different from old
            choices = [v for v in [-1, 0, 1] if v != old_val]
            new_val = random.choice(choices)
            
            delta = (new_val - old_val) * weights[i]
            new_sum = current_sum + delta
            
            if new_sum == 0:
                T *= cooling_rate
                continue
            
            if abs(new_sum) < abs(current_sum):
                # Always accept improvement
                current[i] = new_val
                current_sum = new_sum
                if run_best_sum is None or abs(current_sum) < abs(run_best_sum):
                    run_best_sum = current_sum
                    run_best_coeffs = current[:]
            else:
                # Worse solution: accept with probability exp(-delta_abs / T)
                delta_abs = abs(new_sum) - abs(current_sum)
                if delta_abs < 1000000000:
                    prob = exp(-delta_abs / T)
                    if random.random() < prob:
                        current[i] = new_val
                        current_sum = new_sum
                        if run_best_sum is None or abs(current_sum) < abs(run_best_sum):
                            run_best_sum = current_sum
                            run_best_coeffs = current[:]
            
            T *= cooling_rate
        
        # Final local search to polish
        if current_sum != 0:
            polished, polished_sum = local_search(current, current_sum)
            if run_best_sum is None or abs(polished_sum) < abs(run_best_sum):
                run_best_sum = polished_sum
                run_best_coeffs = polished[:]
        else:
            # Find best neighbor of 0
            for j in range(n):
                old_val = current[j]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    ns = (new_val - old_val) * weights[j]
                    if ns != 0:
                        if run_best_sum is None or abs(ns) < abs(run_best_sum):
                            run_best_sum = ns
                            run_best_coeffs = current[:]
                            run_best_coeffs[j] = new_val
            if run_best_coeffs is not None:
                polished, polished_sum = local_search(run_best_coeffs, run_best_sum)
                if abs(polished_sum) < abs(run_best_sum):
                    run_best_sum = polished_sum
                    run_best_coeffs = polished[:]
        
        # Update global best
        if run_best_sum is not None:
            if best_sum is None or abs(run_best_sum) < abs(best_sum):
                best_sum = run_best_sum
                best_coeffs = run_best_coeffs[:]
    
    # Run multiple independent annealing runs
    while time.time() < deadline:
        simulated_annealing()
        
        # Early exit if we found a very small sum
        if best_sum is not None and abs(best_sum) <= 1:
            break
    
    if best_coeffs is None:
        # Fallback: 1/2 - 1/3 = 1/6
        fallback = [0] * 30
        fallback[1] = 1   # 1/2
        fallback[2] = -1  # -1/3
        return fallback
    
    return best_coeffs
