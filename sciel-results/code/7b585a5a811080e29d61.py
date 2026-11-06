import math
import numpy as np
import time
from functools import reduce
from collections import defaultdict

def construct():
    start_time = time.time()
    n = 30
    
    # Calculate LCM of 1..30
    H = reduce(math.lcm, range(1, n + 1))
    
    # Weights W_i = H / i
    W = [H // i for i in range(1, n + 1)]
    
    # Beam Search
    # State: (sum_value, coeffs_tuple)
    # We keep the K states with smallest absolute sum_value
    K = 5000
    
    # Initial state
    # sum=0, coeffs=()
    # Use a list of tuples (sum, coeffs)
    # To avoid storing full coeffs for all K, we can store path or just coeffs.
    # Storing coeffs (list of 30 ints) for 5000 states is fine.
    
    current_states = [(0, [0]*n)]
    
    for i in range(n):
        next_states = []
        w = W[i]
        
        for s, coeffs in current_states:
            for c in [-1, 0, 1]:
                new_s = s + c * w
                new_coeffs = coeffs.copy()
                new_coeffs[i] = c
                next_states.append((new_s, new_coeffs))
        
        # Sort by absolute value of sum
        # We want to keep the K smallest |sum|
        # But we must ensure we don't lose diversity?
        # For harmonic sums, smallest |sum| is the primary objective.
        next_states.sort(key=lambda x: abs(x[0]))
        
        # Keep top K
        current_states = next_states[:K]
        
        # Time check
        if time.time() - start_time > 50:
            break
            
    # Find the best non-zero sum
    best_val = float('inf')
    best_coeffs = None
    
    for s, coeffs in current_states:
        if s != 0:
            if abs(s) < best_val:
                best_val = abs(s)
                best_coeffs = coeffs
                
    # If we found a very small sum (e.g., 1), we are done.
    # Otherwise, try Local Search to improve.
    
    if best_coeffs is None:
        # Fallback: Greedy
        coeffs_greedy = [0] * n
        curr_sum = 0
        for i in range(n):
            w = W[i]
            best_c = 0
            best_val_local = curr_sum
            for c in [-1, 0, 1]:
                val = curr_sum + c * w
                if abs(val) < abs(best_val_local):
                    best_val_local = val
                    best_c = c
            coeffs_greedy[i] = best_c
            curr_sum = best_val_local
        if curr_sum == 0:
            # Force non-zero
            # Flip the last coefficient
            if coeffs_greedy[-1] == 1:
                coeffs_greedy[-1] = -1
            else:
                coeffs_greedy[-1] = 1
        return coeffs_greedy
        
    # Local Search to see if we can get 1
    # If best_val > 1, try to reduce it
    if best_val > 1:
        current_coeffs = best_coeffs
        current_sum = sum(c * W[i] for i, c in enumerate(current_coeffs))
        best_local_val = abs(current_sum)
        best_local_coeffs = current_coeffs.copy()
        
        no_improve_steps = 0
        max_steps = 5000
        step = 0
        
        while step < max_steps and no_improve_steps < 500:
            # Pick random index
            idx = np.random.randint(n)
            old_c = current_coeffs[idx]
            # Pick new value different from old
            choices = [c for c in [-1, 0, 1] if c != old_c]
            new_c = choices[np.random.randint(len(choices))]
            
            delta = (new_c - old_c) * W[idx]
            new_sum = current_sum + delta
            
            if new_sum != 0:
                new_val = abs(new_sum)
                if new_val < best_local_val:
                    current_coeffs[idx] = new_c
                    current_sum = new_sum
                    best_local_val = new_val
                    best_local_coeffs = current_coeffs.copy()
                    no_improve_steps = 0
                else:
                    # Accept with small probability if close? (Simulated Annealing)
                    # For now, just hill climbing
                    no_improve_steps += 1
            else:
                # If new_sum is 0, it's a valid intermediate but we need non-zero final
                # If we land on 0, we can try to move away to a small non-zero
                # But hill climbing usually rejects 0 if we want min non-zero
                # Actually, passing through 0 is good.
                # If we hit 0, we can try to add a small weight to get 1?
                # For simplicity, reject 0 in this local search phase, 
                # unless we are stuck.
                no_improve_steps += 1
                
            step += 1
            if time.time() - start_time > 55:
                break
                
        if best_local_val < best_val:
            best_val = best_local_val
            best_coeffs = best_local_coeffs

    return best_coeffs
