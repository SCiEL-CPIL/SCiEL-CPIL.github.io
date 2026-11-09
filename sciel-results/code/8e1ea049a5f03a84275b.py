import time
import random
from math import gcd
from functools import reduce

def _lcm_range(n):
    result = 1
    for i in range(2, n + 1):
        result = result * i // gcd(result, i)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin
    
    n = 30
    lcm_val = _lcm_range(n)
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    # Best solution found so far
    best_S = float('inf')
    best_a = None
    
    # Initial seed: 1/29 - 1/30
    # a[28] = 1, a[29] = -1
    init_a = [0] * n
    init_a[28] = 1
    init_a[29] = -1
    init_S = init_a[28] * weights[28] - init_a[29] * weights[29] # 1*w29 - 1*w30? No, weights[i] = L/(i+1) in 0-index?
    # Wait, weights[i] corresponds to 1/(i+1).
    # weights[28] = L/29. weights[29] = L/30.
    # Sum = 1*L/29 - 1*L/30.
    init_S = weights[28] - weights[29]
    
    if abs(init_S) < best_S and init_S != 0:
        best_S = abs(init_S)
        best_a = init_a[:]
        
    # Simulated Annealing
    # We will run multiple independent SA runs
    num_runs = 20
    # Adjust steps per run based on time budget
    # Let's estimate steps per second. A simple loop in Python is ~1e6-1e7 ops/sec depending on complexity.
    # Our step involves a few integer ops and a random choice. Let's assume ~1e5-1e6 steps/sec.
    # We have ~70 seconds. Total budget ~ 5e6 - 5e7 steps.
    # Let's try 100,000 steps per run * 20 runs = 2,000,000 steps. This is safe.
    
    steps_per_run = 100000
    
    for run in range(num_runs):
        if time.time() > deadline - 5:
            break
            
        # Random restart
        a = [0] * n
        # Start with a random sparse configuration to diversify
        # Or start from best_a? Multi-start usually implies randomization.
        # Let's mix: 50% random, 50% from best
        if random.random() < 0.5 and best_a:
            a = best_a[:]
            # Perturb
            for _ in range(5):
                idx = random.randint(0, n - 1)
                a[idx] = random.choice([-1, 0, 1])
        else:
            # Random sparse
            num_nonzero = random.randint(2, 6)
            positions = random.sample(range(n), num_nonzero)
            for p in positions:
                a[p] = random.choice([-1, 1])
                
        S = sum(a[i] * weights[i] for i in range(n))
        if S == 0:
            # Force non-zero
            a[random.randint(0, n-1)] = 1
            S = weights[a.index(1)] # This is wrong, need to recalc or update properly
            # Just recalculate for simplicity in init
            S = sum(a[i] * weights[i] for i in range(n))
            
        current_S = S
        current_energy = abs(S)
        
        # Cooling schedule
        T_start = 10000
        T_end = 1
        alpha = (T_end / T_start) ** (1.0 / steps_per_run)
        
        T = T_start
        
        for step in range(steps_per_run):
            if time.time() > deadline - 2:
                break
                
            # Pick random index
            idx = random.randint(0, n - 1)
            old_val = a[idx]
            
            # Pick new value
            # Uniform from {-1, 0, 1} excluding old_val
            # To avoid list allocation, use random choice from precomputed or logic
            # Possible values:
            if old_val == 0:
                new_val = random.choice([-1, 1])
            else:
                # old_val is 1 or -1
                # choices: 0 and -old_val
                if random.random() < 0.5:
                    new_val = 0
                else:
                    new_val = -old_val
            
            if new_val == old_val:
                continue
                
            delta = (new_val - old_val) * weights[idx]
            new_S = current_S + delta
            
            if new_S == 0:
                # Reject zero sum
                # In SA, we might want to allow passing through 0? No, 0 is a "hole" in our energy landscape.
                # If we reject, we might get stuck.
                # However, the probability of hitting exactly 0 is low.
                continue
            
            new_energy = abs(new_S)
            delta_E = new_energy - current_energy
            
            # Acceptance probability
            if delta_E < 0 or random.random() < pow(2.718281828, -delta_E / T):
                # Accept
                a[idx] = new_val
                current_S = new_S
                current_energy = new_energy
                
                if current_energy < best_S:
                    best_S = current_energy
                    best_a = a[:]
            
            T *= alpha
    
    # Final polish: Local search on the best found
    if best_a is not None and time.time() < deadline - 1:
        a = best_a[:]
        S = sum(a[i] * weights[i] for i in range(n))
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for idx in range(n):
                old_val = a[idx]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * weights[idx]
                    new_S = S + delta
                    if new_S != 0 and abs(new_S) < best_S:
                        a[idx] = new_val
                        S = new_S
                        best_S = abs(S)
                        best_a = a[:]
                        improved = True
                        break # Restart loop since S changed significantly? Or just continue?
                        # Restarting the for loop is safer to check all indices with new S
                        # But breaking inner and letting while loop restart is fine.

    return best_a
