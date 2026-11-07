"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

def objective(h):
    """
    Compute the evaluation value:
    value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    """
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / len(h) * 2

def construct():
    """
    Uses random restarts with local hill climbing to minimize the maximum
    correlation between h and 1-h.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Pre-compute 1-h for speed? No, h changes. 
    # But we can optimize the objective calculation.
    
    def local_search(h_init, max_iters=5000):
        h = h_init.copy()
        current_val = objective(h)
        # Simple coordinate descent or random perturbations
        # Since N=100, random perturbations of a few elements are fast.
        
        # Use a simple stochastic hill climbing
        # Step size starts moderate and decays
        step_size = 0.1
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random index to perturb
            idx = np.random.randint(N)
            old_val = h[idx]
            new_val = np.clip(old_val + np.random.uniform(-step_size, step_size), 0.0, 1.0)
            
            if new_val == old_val:
                continue
                
            h[idx] = new_val
            
            # Enforce sum constraint approximately?
            # The constraint is sum(h) = N/2 within tolerance.
            # If we perturb, we break the sum.
            # We need to correct the sum.
            # Simple correction: distribute the error equally? Or adjust one other element?
            # Better: Project back to the hyperplane sum(h) = N/2.
            # Error = sum(h) - N/2
            # h -= error / N
            # This maintains the mean but might push values out of [0,1].
            # If out of [0,1], we clip and re-project? This is complex.
            
            # Alternative: Only accept moves that keep sum close?
            # Or, simply re-normalize? No, normalization changes the shape.
            
            # Let's stick to a simpler constraint handling:
            # After perturbation, if sum is off, adjust the element with the most "room" or just 
            # accept the move if the sum error is small, otherwise reject.
            # Actually, the evaluator requires sum(h) to be N/2 within 1e-9*N.
            # So we MUST enforce this strictly.
            
            # Strategy: Perturb h[idx], then adjust h[j] (j != idx) to maintain sum.
            # Choose j randomly.
            if idx == 0:
                j = 1
            else:
                j = 0 # Or random. Let's use random.
            
            # Let's pick a random j != idx
            j = np.random.randint(N)
            if j == idx:
                j = (j + 1) % N
            
            # We want h[idx] -> new_val
            # We need h[j] -> old_h_j + (old_val - new_val)
            old_h_j = h[j]
            new_h_j = old_h_j + (old_val - new_val)
            
            if new_h_j < 0.0 or new_h_j > 1.0:
                # Revert
                h[idx] = old_val
                continue
                
            h[idx] = new_val
            h[j] = new_h_j
            
            new_val_obj = objective(h)
            
            if new_val_obj < current_val:
                current_val = new_val_obj
                # Decay step size
                step_size *= 0.999
            else:
                # Revert
                h[idx] = old_val
                h[j] = old_h_j
                # Increase step size slightly if stuck?
                step_size = min(0.2, step_size * 1.01)
                
        return h, current_val

    # Run multiple restarts
    num_restarts = 0
    while time.time() - start_time < time_limit:
        # Start from a random valid point
        # Generate random h, then project to sum=N/2 and [0,1]
        h_rand = np.random.uniform(0, 1, N)
        # Project to sum = N/2
        # Simple projection: shift all elements
        target_sum = N / 2.0
        current_sum = np.sum(h_rand)
        h_rand += (target_sum - current_sum) / N
        
        # Clip to [0,1]
        h_rand = np.clip(h_rand, 0.0, 1.0)
        
        # Re-project to sum = N/2 after clipping?
        # Clipping might change the sum.
        # Iterative projection:
        for _ in range(10):
            h_rand = np.clip(h_rand, 0.0, 1.0)
            current_sum = np.sum(h_rand)
            diff = target_sum - current_sum
            if abs(diff) < 1e-12:
                break
            # Distribute diff to elements not at bounds
            # Simple approach: add diff/N to all, then clip again
            h_rand += diff / N
            
        # Check validity
        if not np.all(np.isfinite(h_rand)) or np.sum(h_rand) < target_sum - 1e-9 or np.sum(h_rand) > target_sum + 1e-9:
            # Fallback to constant
            h_rand = np.full(N, 0.5)
            
        h_new, val_new = local_search(h_rand)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            
        num_restarts += 1
        
        # If we've found a very good value, maybe stop?
        # The theoretical minimum is likely 0.5 (constant). 
        # If we can't beat 0.5, we return 0.5.
        # If we can, we keep going.
        
        # Check if we are at the constant solution
        if best_val <= 0.5 + 1e-9:
            break

    # Final safety check: return constant if we didn't find anything better
    # or if the best found is invalid
    if best_val > objective(np.full(N, 0.5)):
        return np.full(N, 0.5)
        
    return best_h
# EVOLVE-BLOCK-END
