import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    # The evaluator kills after 90s, but the prompt says 60s limit for evaluation.
    # We will use a conservative deadline of 58s to be safe.
    deadline = start_time + 58.0
    
    # Precompute LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // i for i in range(1, 31)]
    
    # Target: Find coefficients c in {-1, 0, 1} such that sum(c[i]*weights[i]) != 0
    # and |sum| is minimized. The minimum possible nonzero absolute value is 1.
    # If we find a sum of 1 or -1, we can stop immediately as it's optimal.
    
    best_c = None
    best_abs = float('inf')
    
    # Helper: Greedy descent from a given state
    # State: current_sum, current_c
    # Strategy: While sum != 0, find a weight w such that adding/subtracting it 
    # reduces |sum|. Specifically, if sum > 0, we want to subtract a weight w 
    # such that w < sum (to avoid crossing zero) OR w is close to sum.
    # Actually, a better greedy: 
    # If sum > 0, we want to make it smaller. We can change c[i] from 1 to 0 (subtract w[i]) 
    # or from 0 to -1 (subtract w[i]) or from -1 to 0 (add w[i], bad).
    # So we look for available "subtractions" w[i] (where c[i] is 0 or 1) that are <= current_sum.
    # We pick the largest such w[i] to reduce sum most efficiently.
    # If no such w[i] exists, we might need to accept a worse move or restart.
    
    def greedy_descent(c, current_sum):
        # Make a copy
        c = c[:]
        s = current_sum
        if s == 0:
            return c, 0
        
        # Sort indices by weight descending for faster search? 
        # Or just iterate. 30 items is small.
        # We need to find i such that:
        # if s > 0:
        #   we can decrease s by w[i] if c[i] is 1 (change to 0) or 0 (change to -1).
        #   We want the largest w[i] <= s.
        # if s < 0:
        #   we can increase s by w[i] (make it less negative) if c[i] is -1 (change to 0) or 0 (change to 1).
        #   We want the largest w[i] <= |s|.
        
        changed = True
        while changed and s != 0:
            changed = False
            target_abs = abs(s)
            best_w = 0
            best_idx = -1
            best_new_val = 0
            
            for i in range(30):
                w = weights[i]
                if w > target_abs:
                    continue
                # Can we apply this weight to reduce |s|?
                if s > 0:
                    # We want to subtract w.
                    # If c[i] == 1, changing to 0 subtracts w.
                    # If c[i] == 0, changing to -1 subtracts w.
                    if c[i] == 1 or c[i] == 0:
                        if w > best_w:
                            best_w = w
                            best_idx = i
                            best_new_val = 0 if c[i] == 1 else -1
                else: # s < 0
                    # We want to add w (increase s towards 0).
                    # If c[i] == -1, changing to 0 adds w.
                    # If c[i] == 0, changing to 1 adds w.
                    if c[i] == -1 or c[i] == 0:
                        if w > best_w:
                            best_w = w
                            best_idx = i
                            best_new_val = 0 if c[i] == -1 else 1
            
            if best_idx != -1:
                # Apply the move
                old_val = c[best_idx]
                delta = (best_new_val - old_val) * weights[best_idx]
                s += delta
                c[best_idx] = best_new_val
                changed = True
        
        return c, s

    # Multi-start ILS
    # We will try many random initializations and run greedy descent.
    # If greedy descent gets stuck (s != 0 and no valid greedy move), we perturb.
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Random initialization
        # Bias towards sparsity? Or random?
        # Let's try a mix.
        if random.random() < 0.5:
            c = [random.choice([-1, 0, 1]) for _ in range(30)]
        else:
            # Sparse init
            c = [0] * 30
            for _ in range(random.randint(1, 5)):
                idx = random.randint(0, 29)
                c[idx] = random.choice([-1, 1])
        
        current_sum = sum(c[i] * weights[i] for i in range(30))
        
        if current_sum == 0:
            # Perturb to get nonzero
            idx = random.randint(0, 29)
            if c[idx] == 0:
                c[idx] = random.choice([-1, 1])
            else:
                c[idx] = 0 # This might make it 0 again, but let's see
            current_sum = sum(c[i] * weights[i] for i in range(30))
            if current_sum == 0:
                # Force a 1
                c[0] = 1
                current_sum = sum(c[i] * weights[i] for i in range(30))
        
        # Run Greedy Descent
        c_desc, s_desc = greedy_descent(c, current_sum)
        
        if abs(s_desc) < best_abs:
            best_abs = abs(s_desc)
            best_c = c_desc[:]
            if best_abs == 1:
                return best_c
        
        # If we didn't reach 1, try to perturb the local minimum and descend again
        # Perturbation: flip 1-3 random bits
        for _ in range(10): # Try 10 perturbations per restart
            if time.time() > deadline:
                break
            
            c_pert = c_desc[:]
            s_pert = s_desc
            
            # Flip 1 to 3 random indices
            num_flips = random.randint(1, 3)
            for _ in range(num_flips):
                idx = random.randint(0, 29)
                old_val = c_pert[idx]
                new_val = random.choice([-1, 0, 1])
                if new_val != old_val:
                    delta = (new_val - old_val) * weights[idx]
                    s_pert += delta
                    c_pert[idx] = new_val
            
            if s_pert == 0:
                continue
                
            # Greedy descent on perturbed state
            c_final, s_final = greedy_descent(c_pert, s_pert)
            
            if abs(s_final) < best_abs:
                best_abs = abs(s_final)
                best_c = c_final[:]
                if best_abs == 1:
                    return best_c

    # Return best found
    if best_c is None:
        # Fallback: simple nonzero
        return [1] + [0]*29
    return best_c
