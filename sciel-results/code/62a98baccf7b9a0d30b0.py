import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Global state
        best_x = None
        best_f = np.inf
        eval_count = 0
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise BudgetExceeded
            f_val = func(x)
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Strategic Multi-Start Sampling
        # Lunacek bi-Rastrigin has a global minimum near the origin (after shift/rotation)
        # and a secondary local minimum. 
        # We sample a few points to identify the best basin to start the intensive search.
        
        # Strategic candidates:
        # 1. Origin (often close to global min in standard Lunacek)
        # 2. A point in the "other" basin region (heuristic: shifted by ~4-5 units)
        # 3. Random points to cover the space
        
        candidates = []
        candidates.append(np.zeros(n))
        
        # Heuristic for secondary basin: In standard Lunacek, the secondary min is 
        # often around x_i = 2.5 * sign(x_i) for some pattern, or shifted. 
        # Since we don't know the exact shift/rotation, we try a few diverse points.
        # A common feature of Lunacek is the 'd' parameter creating a secondary well.
        # Let's try points with mixed signs and magnitudes.
        
        # Point 2: Shifted in one direction
        p2 = np.zeros(n)
        p2[0] = 4.0
        candidates.append(p2)
        
        # Point 3: Shifted in another direction
        p3 = np.zeros(n)
        p3[1] = -4.0
        candidates.append(p3)
        
        # Points 4-8: Random
        for _ in range(5):
            candidates.append(rng.uniform(-5, 5, n))
            
        for x in candidates:
            if time.time() > deadline or eval_count >= budget:
                break
            safe_func(x)
            
        if best_x is None:
            return np.zeros(n), 0.0

        # Phase 2: Adaptive (1+1)-ES from the best found point
        # We use the 1/5th success rule to adapt the step size.
        # This is more robust than fixed step decay for multimodal landscapes 
        # if we restart, but here we run from the best initial point.
        
        current_x = best_x.copy()
        current_f = best_f
        
        # Initial step size: large enough to explore, small enough to converge
        # Given domain [-5, 5], a step of 1.0 is reasonable.
        step_size = 1.0
        
        # 1/5th success rule parameters
        # c_sigma is the adaptation rate. Typically around 1/5 to 1/3.
        # We want to adapt quickly.
        c_sigma = 0.5 
        
        # We will run this for the remaining budget, but we need to be careful 
        # about getting stuck in a local minimum.
        # The (1+1)-ES is a local optimizer. 
        # To handle the bi-modal nature, we might need to restart if stagnation occurs.
        
        # Let's implement a simple restart strategy:
        # If no improvement for K evaluations, restart from a random point 
        # or a point near the origin/secondary basin if not already there.
        
        stagnation_count = 0
        max_stagnation = 20  # If no improvement in 20 evals, consider restart
        restart_count = 0
        max_restarts = 3  # Limit restarts to save budget for polish
        
        # We'll use a loop until budget is exhausted or time runs out
        while eval_count < budget and time.time() < deadline:
            # Generate a candidate
            # Normal perturbation
            if step_size < 1e-8:
                break
                
            delta = rng.normal(0, step_size, n)
            x_new = current_x + delta
            x_new = np.clip(x_new, -5, 5)
            
            f_new = safe_func(x_new)
            
            # Adapt step size based on success
            # Success if f_new < current_f
            if f_new < current_f:
                current_x = x_new
                current_f = f_new
                # Increase step size slightly (success)
                step_size *= (1 + c_sigma)
                stagnation_count = 0
            else:
                # Decrease step size (failure)
                step_size *= (1 - c_sigma)
                stagnation_count += 1
                
            # Check for stagnation and potential restart
            if stagnation_count >= max_stagnation and restart_count < max_restarts:
                # Try a restart
                # Pick a new starting point
                # Heuristic: Try the other major basin or a random point
                if restart_count % 2 == 0:
                    # Try origin neighborhood
                    x_restart = rng.normal(0, 0.5, n)
                else:
                    # Try random
                    x_restart = rng.uniform(-5, 5, n)
                    
                f_restart = safe_func(x_restart)
                
                if f_restart < current_f:
                    current_x = x_restart
                    current_f = f_restart
                    step_size = 1.0 # Reset step size for new basin
                    stagnation_count = 0
                    restart_count += 1
                else:
                    # If restart didn't help, keep going but reset stagnation counter
                    # to avoid infinite restart loop on same bad points
                    stagnation_count = 0
                    restart_count += 1
                    # Optionally shrink step size more aggressively to try to escape
                    step_size *= 0.5
                    
            # Safety check: if step size is too small, we've converged
            if step_size < 1e-6:
                break

        # Phase 3: Final Polish
        # If we have budget left, do a very fine-grained search around the best point
        if eval_count < budget - 5 and time.time() < deadline:
            polish_step = 0.01
            for _ in range(10):
                if time.time() > deadline or eval_count >= budget:
                    break
                delta = rng.normal(0, polish_step, n)
                x_new = best_x + delta
                x_new = np.clip(x_new, -5, 5)
                f_new = safe_func(x_new)
                if f_new < best_f:
                    best_x = x_new
                    best_f = f_new
                # Shrink polish step
                polish_step *= 0.8

        return best_x, best_f

    def _cov_to_b_d(self, C, n):
        # Not used in this strategy, but kept for interface compatibility if needed
        try:
            L = np.linalg.cholesky(C)
            return L, np.ones(n)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs, np.sqrt(eigvals)
