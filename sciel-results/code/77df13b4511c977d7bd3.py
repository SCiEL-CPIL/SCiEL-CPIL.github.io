import numpy as np
import time
import random

N = 100

def compute_value(h):
    # h is a numpy array of length N
    # Calculate correlation of h and 1-h
    # np.correlate(h, 1-h, mode='full')
    # The value is max(corr) / N * 2
    corr = np.correlate(h, 1 - h, mode='full')
    return np.max(corr) / N * 2

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # We will try a few random initializations and local searches
    # Since N=100 is small, we can do many iterations.
    
    # Helper for local search on a given h
    def local_search(h, max_iters=20000):
        current_h = h.copy()
        current_val = compute_value(current_h)
        
        # Pre-compute 1-h for efficiency? No, just recompute.
        # N=100, correlate is fast.
        
        no_improve_count = 0
        last_improve_time = time.time()
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if time.time() - last_improve_time > 5.0:
                # If stuck for 5s, break to try another restart
                break
                
            # Pick two random distinct indices
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            
            # Swap h[i] and h[j]
            # We can restrict to swapping values that are different to ensure change
            if current_h[i] == current_h[j]:
                continue
                
            orig_i = current_h[i]
            orig_j = current_h[j]
            
            current_h[i] = orig_j
            current_h[j] = orig_i
            
            new_val = compute_value(current_h)
            
            if new_val < current_val:
                current_val = new_val
                no_improve_count = 0
                last_improve_time = time.time()
            else:
                # Revert
                current_h[i] = orig_i
                current_h[j] = orig_j
                no_improve_count += 1
                if no_improve_count > 5000:
                    break
                    
        return current_h, current_val

    # Try 10 restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with 50 ones and 50 zeros
        h = np.zeros(N)
        ones_indices = random.sample(range(N), N // 2)
        h[ones_indices] = 1.0
        
        h, val = local_search(h, max_iters=5000)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Optional: Polish with SLSQP if time permits and we have a good candidate
        # The prompt mentions SLSQP polish improves things.
        # Let's add a simple polish step for the best found so far in each restart?
        # Or just do it at the end for the global best.
        
    # Final Polish with scipy
    if best_h is not None and time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            
            # We want to minimize max(corr).
            # To make it smooth, we can use a soft-max or just minimize the sum of squares of the top few correlations?
            # Or simply minimize the max by iterating.
            # Let's try minimizing the raw max with L-BFGS-B, it might work well enough.
            # However, max is non-differentiable.
            # A common trick: minimize sum(exp(beta * corr_k)) / beta for large beta.
            # Or just use the discrete search result, which is often very strong for this specific metric.
            
            # Given the time limit and the robustness of the swap search, 
            # and the fact that continuous values might not help much more than 0/1 
            # for "max overlap" (since 0/1 minimizes the self-term to 0),
            # I will stick to the discrete result unless the polish is very fast.
            
            # Let's try a quick SLSQP polish on the best_h
            # Objective: max correlation
            # This is hard for SLSQP directly.
            # I will skip the complex polish to avoid instability and rely on the strong discrete search.
            # The knowledge says "Direct 100-dim DE ... SLSQP polish ... 1.291 -> 1.308".
            # This implies continuous search is better.
            
            # Let's implement a simple coordinate descent polish.
            # For each i, try changing h[i] slightly.
            # This is O(N * steps).
            
            h_polish = best_h.copy()
            val_polish = best_val
            step_size = 0.05
            for _ in range(200): # 200 passes
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5.0:
                        break
                    for delta in [-step_size, step_size]:
                        new_h = h_polish.copy()
                        new_val_i = h_polish[i] + delta
                        if new_val_i < 0: new_val_i = 0
                        if new_val_i > 1: new_val_i = 1
                        new_h[i] = new_val_i
                        
                        # Check sum constraint? 
                        # The task says sum(h) must be N/2.
                        # Coordinate descent that changes one value breaks the sum.
                        # We need to move another value to compensate, or project back.
                        # Projecting back is complex.
                        # Better: Swap-based search maintains sum exactly.
                        # Continuous search needs to maintain sum.
                        # Let's do "swap-continuous": pick i, j. Change h[i] by +d, h[j] by -d.
                        
                        j = random.randint(0, N - 1)
                        if i == j: continue
                        
                        d = step_size
                        # Ensure bounds
                        h_i_new = h_polish[i] + d
                        h_j_new = h_polish[j] - d
                        if h_i_new < 0 or h_i_new > 1 or h_j_new < 0 or h_j_new > 1:
                            # Try smaller d or skip
                            continue
                        
                        new_h[i] = h_i_new
                        new_h[j] = h_j_new
                        
                        new_val = compute_value(new_h)
                        if new_val < val_polish:
                            val_polish = new_val
                            h_polish = new_h
                            improved = True
                            break # Restart i loop? Or continue?
                    if improved:
                        break # Restart i loop
                if not improved:
                    step_size *= 0.9
                    if step_size < 1e-4:
                        break
                        
            if val_polish < best_val:
                best_val = val_polish
                best_h = h_polish
                
        except Exception:
            pass

    return best_h

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 100

def compute_value(h):
    corr = np.correlate(h, 1 - h, mode='full')
    return np.max(corr) / N * 2

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_h = None
    best_val = float('inf')
    
    def local_search(h, max_iters=10000):
        current_h = h.copy()
        current_val = compute_value(current_h)
        no_improve_count = 0
        last_improve_time = time.time()
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if time.time() - last_improve_time > 3.0:
                break
                
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            
            if current_h[i] == current_h[j]:
                continue
                
            orig_i = current_h[i]
            orig_j = current_h[j]
            
            current_h[i] = orig_j
            current_h[j] = orig_i
            
            new_val = compute_value(current_h)
            
            if new_val < current_val:
                current_val = new_val
                no_improve_count = 0
                last_improve_time = time.time()
            else:
                current_h[i] = orig_i
                current_h[j] = orig_j
                no_improve_count += 1
                if no_improve_count > 2000:
                    break
                    
        return current_h, current_val

    # Try a few restarts
    for r in range(5):
        if time.time() > deadline:
            break
            
        h = np.zeros(N)
        ones_indices = random.sample(range(N), N // 2)
        h[ones_indices] = 1.0
        
        h, val = local_search(h, max_iters=5000)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Polish with continuous swaps
    if best_h is not None and time.time() < deadline - 10.0:
        h_polish = best_h.copy()
        val_polish = best_val
        step_size = 0.1
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                for delta in [-step_size, step_size]:
                    j = random.randint(0, N - 1)
                    if i == j:
                        continue
                    h_i_new = h_polish[i] + delta
                    h_j_new = h_polish[j] - delta
                    if h_i_new < 0 or h_i_new > 1 or h_j_new < 0 or h_j_new > 1:
                        continue
                    new_h = h_polish.copy()
                    new_h[i] = h_i_new
                    new_h[j] = h_j_new
                    new_val = compute_value(new_h)
                    if new_val < val_polish:
                        val_polish = new_val
                        h_polish = new_h
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                step_size *= 0.8
                if step_size < 1e-3:
                    break
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish

    return best_h
# EVOLVE-BLOCK-END
