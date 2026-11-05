import time
import random
import math
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 60.0  # 60 seconds limit, keep safety margin
    
    # 1. Calculate LCM of 1..30
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
        
    L = reduce(lcm, range(1, 31))
    
    # Weights W[i] = L / (i+1)
    W = [L // (i + 1) for i in range(30)]
    
    best_abs_val = float('inf')
    best_config = None
    best_val = 0
    
    def eval_config(c):
        return sum(c[i] * W[i] for i in range(30))
    
    # Random restarts with Simulated Annealing
    # Number of restarts depends on time
    max_restarts = 50
    
    for r in range(max_restarts):
        if time.time() - start_time > timeout - 5.0:
            break
            
        # Initialize random config
        c = [random.choice([-1, 0, 1]) for _ in range(30)]
        cur_val = eval_config(c)
        
        if cur_val != 0:
            if abs(cur_val) < best_abs_val:
                best_abs_val = abs(cur_val)
                best_config = c[:]
                best_val = cur_val
                
        # If we found 1, we can't do better (unless we find 0 which is invalid)
        if best_abs_val == 1:
            break
            
        # Simulated Annealing
        T = 1000.0
        T_min = 0.1
        cooling = 0.95
        steps_per_restart = 20000
        
        for step in range(steps_per_restart):
            if time.time() - start_time > timeout - 5.0:
                break
                
            # Pick random index
            idx = random.randint(0, 29)
            old_c = c[idx]
            
            # Pick new value different from old
            options = [v for v in [-1, 0, 1] if v != old_c]
            new_c = random.choice(options)
            
            delta = (new_c - old_c) * W[idx]
            new_val = cur_val + delta
            
            if new_val == 0:
                # Reject zero sum
                continue
                
            new_abs = abs(new_val)
            cur_abs = abs(cur_val)
            
            if new_abs < cur_abs:
                # Accept improvement
                c[idx] = new_c
                cur_val = new_val
                
                if new_abs < best_abs_val:
                    best_abs_val = new_abs
                    best_config = c[:]
                    best_val = cur_val
                    
                    if best_abs_val == 1:
                        break
            else:
                # Accept worse with probability
                diff = new_abs - cur_abs
                if T > 0:
                    p = math.exp(-diff / T)
                    if random.random() < p:
                        c[idx] = new_c
                        cur_val = new_val
                        
            T *= cooling
            if T < T_min:
                T = T_min
                
        if best_abs_val == 1:
            break

    if best_config is None:
        # Fallback to seed if no better found
        return [0] * 28 + [1, -1]
        
    return best_config
