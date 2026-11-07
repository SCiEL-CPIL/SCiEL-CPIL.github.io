import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        # Initial point: center of the box
        x = np.zeros(dim)
        
        # Initial step sizes per dimension
        # Start with a step size that covers a significant portion of the domain
        steps = np.full(dim, 2.0)
        
        # Track best
        fx = func(x)
        calls_made += 1
        best_x = x.copy()
        best_f = fx
        
        # Coordinate descent parameters
        # We will perform a sequence of line searches along each dimension.
        # If no improvement is found after a full cycle of dimensions, we jitter.
        
        # Golden section search parameters
        gr = (np.sqrt(5) - 1) / 2  # ~0.618
        
        # State for line search
        # We maintain a current step size for each dimension.
        
        # Main loop
        # We structure the search in "phases".
        # Phase 1: Global exploration via coordinate descent with large steps.
        # Phase 2: Local exploitation with smaller steps.
        # We adapt step sizes based on success.
        
        # To ensure we use the budget effectively and don't get stuck,
        # we limit the number of function evaluations per line search.
        max_evals_per_line_search = 10
        
        # We will iterate through dimensions repeatedly.
        # If a dimension yields improvement, we shrink its step size slightly (exploitation).
        # If not, we might keep it or expand it if we suspect we are far from optimum?
        # For Schwefel, the landscape is multimodal. Coordinate descent can get stuck.
        # Jittering is key.
        
        # Let's define a helper to perform a 1D minimization along axis k
        # centered at x[k] with step size steps[k].
        # We use a simple hill-climbing with adaptive step within the line search
        # to be robust to the box constraints and non-convexity.
        
        # Actually, a simpler and more robust approach for black-box:
        # Nelder-Mead is often superior for low dimensions (<=10) and small budgets.
        # However, the directive asks for "coordinate-descent hybrid".
        # Let's stick to the directive but make it robust.
        
        # Strategy:
        # 1. Try to improve along each axis using a small number of evals.
        # 2. If an axis improves, accept and reduce step size for that axis.
        # 3. If an axis doesn't improve after trying both directions, increase step size? 
        #    No, increasing step size in a local min is bad. 
        #    Instead, if we make no progress in a full cycle (all dims), we apply a random perturbation.
        
        # Let's refine the "line search":
        # For dimension d:
        #   Try x + steps[d] * e_d and x - steps[d] * e_d.
        #   If one is better, move there.
        #   Then, try to continue in that direction with smaller steps (geometric decay) to find a local minimum along that axis.
        #   If we find a minimum, update x and best.
        #   Reset step size for that dimension based on the distance moved or keep it.
        
        # To handle the "separable structure" hint:
        # Rotated Schwefel is NOT separable. The hint might be misleading or referring to the fact that
        # local gradients often align with axes in transformed spaces? Or maybe it just means
        # that 1D searches are cheap.
        
        # Let's implement a robust coordinate descent with adaptive steps and restarts.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            improved_in_cycle = False
            
            for d in range(dim):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Attempt to improve along dimension d
                # Current value fx
                # Try positive direction
                x_trial = x.copy()
                x_trial[d] += steps[d]
                x_trial = np.clip(x_trial, -5, 5)
                
                # If the trial is identical to x (due to clipping), skip
                if np.array_equal(x_trial, x):
                    continue
                
                fx_trial = func(x_trial)
                calls_made += 1
                
                if fx_trial < fx:
                    # Move to trial
                    x = x_trial
                    fx = fx_trial
                    improved_in_cycle = True
                    
                    # Try to go further in the same direction (hill climbing)
                    # Use a geometric sequence of steps
                    current_step = steps[d]
                    # Limit evaluations for this axis refinement
                    for _ in range(5): # Max 5 additional evals for refinement
                        if calls_made >= budget or time.time() > deadline:
                            break
                        x_trial2 = x.copy()
                        x_trial2[d] += current_step
                        x_trial2 = np.clip(x_trial2, -5, 5)
                        
                        if np.array_equal(x_trial2, x):
                            break
                        
                        fx_trial2 = func(x_trial2)
                        calls_made += 1
                        
                        if fx_trial2 < fx:
                            x = x_trial2
                            fx = fx_trial2
                            current_step *= 0.5 # Shrink step for next try
                        else:
                            # Check negative direction from current best?
                            # Or just stop? 
                            # Let's try the negative direction from the original x before the first move?
                            # No, we are already moved. 
                            # If we can't go further positive, maybe we overshot?
                            # Let's try a step back? No, that's expensive.
                            # Let's just shrink the global step size for this dimension for next cycle.
                            break
                            
                    # Update step size for this dimension
                    # If we moved, we are closer to a local min. Reduce step size.
                    steps[d] *= 0.7 
                    steps[d] = max(steps[d], 1e-4)
                    
                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Positive direction failed. Try negative direction.
                    x_trial_neg = x.copy()
                    x_trial_neg[d] -= steps[d]
                    x_trial_neg = np.clip(x_trial_neg, -5, 5)
                    
                    if np.array_equal(x_trial_neg, x):
                        # Can't move in either direction? Shrink step drastically or jitter?
                        # If we can't move, we are at the boundary or step is too small.
                        # If step is small, we are stuck.
                        steps[d] *= 0.5
                        continue
                    
                    fx_trial_neg = func(x_trial_neg)
                    calls_made += 1
                    
                    if fx_trial_neg < fx:
                        x = x_trial_neg
                        fx = fx_trial_neg
                        improved_in_cycle = True
                        
                        # Hill climbing in negative direction
                        current_step = steps[d]
                        for _ in range(5):
                            if calls_made >= budget or time.time() > deadline:
                                break
                            x_trial2 = x.copy()
                            x_trial2[d] -= current_step
                            x_trial2 = np.clip(x_trial2, -5, 5)
                            
                            if np.array_equal(x_trial2, x):
                                break
                            
                            fx_trial2 = func(x_trial2)
                            calls_made += 1
                            
                            if fx_trial2 < fx:
                                x = x_trial2
                                fx = fx_trial2
                                current_step *= 0.5
                            else:
                                break
                                
                        steps[d] *= 0.7
                        steps[d] = max(steps[d], 1e-4)
                        
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # Neither direction improved.
                        # This suggests we might be at a local minimum along this axis,
                        # or the step size is too large and we jumped over a valley,
                        # or too small.
                        # For now, shrink step size slightly to allow finer search next time.
                        steps[d] *= 0.9
                        steps[d] = max(steps[d], 1e-5)
            
            # End of cycle over all dimensions
            
            if not improved_in_cycle:
                # Stagnation detected: no improvement in any dimension.
                # Apply a random perturbation to escape local minimum.
                # The magnitude of perturbation should be significant.
                # Use a random direction with a magnitude proportional to current step sizes or a fixed value.
                # Let's use a random vector with norm ~ 1.0
                jitter = rng.standard_normal(dim)
                # Scale jitter to be comparable to step sizes
                # Or just use a fixed scale?
                # Let's use a scale of 1.0
                x = x + 1.0 * jitter
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                calls_made += 1
                
                # Update step sizes? 
                # Maybe reset step sizes to a medium value to explore again?
                # Or keep them?
                # If we were stuck, the step sizes might be too small.
                # Let's increase step sizes for dimensions that didn't move?
                # We don't track which ones moved.
                # Let's just reset all step sizes to a medium value, e.g., 1.0
                # This acts as a restart.
                steps = np.full(dim, 1.0)
                
                # Update best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        return best_x, best_f
