import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_fx = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: LHS Warm-Start Exploration ---
        # Use Latin Hypercube Sampling to guarantee diverse coverage of the box
        # before adaptive search begins. This is superior to a single random start
        # or a dense grid which might miss the basin if the shift is large.
        
        explore_budget = min(80, budget // 5)
        
        # Generate LHS points
        # We divide each dimension into explore_budget strata
        # For simplicity and speed, we generate random points within each stratum
        # Or just use a random permutation approach for LHS
        # Standard LHS: for each dimension, partition [0,1] into N intervals, pick one random point per interval, shuffle.
        
        n_lhs = explore_budget
        # Create indices for each dimension
        # For each dimension d, we have n_lhs intervals [i/n_lhs, (i+1)/n_lhs]
        # We pick a random point in each, then shuffle the order for each dimension independently?
        # No, LHS means: 
        # 1. For each dimension d, create a permutation of 0..N-1
        # 2. The i-th point has coordinate (perm_d[i] + U) / N where U ~ U(0,1)
        
        # Let's implement a simple LHS generator
        # Actually, for dim=5 and N=80, we can just use random points if we want to save code complexity,
        # but the lesson says LHS is better. Let's implement a basic LHS.
        
        # Better: Use scipy.stats.qmc.LatinHypercube if available, but let's stick to numpy for safety.
        # Simple LHS implementation:
        # X is N x D
        # For each dim d:
        #   perm = rng.permutation(N)
        #   X[:, d] = (perm + rng.random(N)) / N
        #   Scale to [-5, 5]
        
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = (perm + rng.random(n_lhs)) / n_lhs
            # Map from [0, 1] to [-5, 5]
            lhs_points[:, d] = lhs_points[:, d] * 10.0 - 5.0
        
        # Evaluate LHS points
        for i in range(n_lhs):
            if calls >= budget or time.time() > deadline:
                break
            safe_call(lhs_points[i])
        
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_fx = safe_call(best_x)

        # --- Phase 2: Adaptive (1+1)-ES Refinement ---
        # Use the best point from exploration as the starting point.
        x = best_x.copy()
        fx = best_fx
        sigma = 1.0
        
        # We use asymmetric scaling: 1.3 up on success, 0.85 down on failure.
        # This was found to exploit the basin more efficiently than symmetric or 1.5/0.87.
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            y = x + sigma * rng.standard_normal(dim)
            fy = safe_call(y)
            
            if fy == np.inf:
                break
                
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.3
            else:
                sigma *= 0.85
                
            # Keep sigma in reasonable range
            if sigma > 4.0:
                sigma = 4.0
            elif sigma < 1e-5:
                sigma = 1e-5

        # --- Phase 3: Deterministic Coordinate-Ascent Polish ---
        # If we have remaining budget and time, perform a precise coordinate ascent.
        # This leverages the "attractive sector" structure where the minimum is often
        # sharp along specific axes.
        
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline - 5.0:
            polish_budget = min(remaining_budget, 50)
            
            # Adaptive step size for coordinate ascent
            step = max(sigma * 0.5, 0.01)
            
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                    
                improved = False
                for d in range(dim):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try moving in positive direction
                    x_trial = best_x.copy()
                    x_trial[d] += step
                    f_trial = safe_call(x_trial)
                    
                    if f_trial == np.inf:
                        break
                        
                    if f_trial < best_fx:
                        improved = True
                    else:
                        # Try moving in negative direction
                        x_trial = best_x.copy()
                        x_trial[d] -= step
                        f_trial = safe_call(x_trial)
                        
                        if f_trial == np.inf:
                            break
                            
                        if f_trial < best_fx:
                            improved = True
                    
                    if improved:
                        step *= 0.8
                        break
                else:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_fx
