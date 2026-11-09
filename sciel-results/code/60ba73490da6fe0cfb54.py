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

        # Phase 1: Global Sampling to find a good starting basin
        # Lunacek has a global and a secondary minimum. 
        # We do a coarse grid or random search to find the best starting point.
        n_samples = 10
        # Use a few strategic points: origin, random points
        candidates = [np.zeros(n)]
        for _ in range(n_samples - 1):
            candidates.append(rng.uniform(-5, 5, n))
        
        for x in candidates:
            if time.time() > deadline or eval_count >= budget:
                break
            safe_func(x)
        
        if best_x is None:
            return np.zeros(n), 0.0

        current_x = best_x.copy()
        current_f = best_f

        # Phase 2: Cyclic Coordinate Descent (CCD) with Global Step Decay
        # Start with a large step size, shrink it geometrically.
        # This mimics the behavior of a robust line search without explicit line searches.
        
        # Parameters for CCD
        initial_step = 2.0
        decay_factor = 0.5
        min_step = 1e-6
        max_cycles = 100  # Safety limit for cycles
        
        step = initial_step
        stagnant_cycles = 0
        max_stagnant = 10  # If no improvement in 10 full cycles, kick or stop

        for cycle in range(max_cycles):
            if time.time() > deadline or eval_count >= budget:
                break
            
            improved_in_cycle = False
            
            for i in range(n):
                if time.time() > deadline or eval_count >= budget:
                    break
                
                # Try positive step
                x_plus = current_x.copy()
                x_plus[i] += step
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = safe_func(x_plus)
                
                if f_plus < current_f:
                    current_x = x_plus
                    current_f = f_plus
                    improved_in_cycle = True
                    # If improved, we could try to go further, but for simplicity 
                    # and to save budget, we accept the step and move to next coordinate.
                    # A more aggressive CCD might do a local line search here.
                    # Given the 500 budget, simple acceptance is efficient.
                    
                    # Optional: Try a larger step if we improved? 
                    # No, let's stick to the global decay for stability.
                else:
                    # Try negative step
                    x_minus = current_x.copy()
                    x_minus[i] -= step
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = safe_func(x_minus)
                    
                    if f_minus < current_f:
                        current_x = x_minus
                        current_f = f_minus
                        improved_in_cycle = True
                    else:
                        # No improvement in this coordinate. 
                        # We could try a smaller step for this coordinate immediately, 
                        # but the global decay handles this over cycles.
                        pass

            if not improved_in_cycle:
                stagnant_cycles += 1
            else:
                stagnant_cycles = 0
                
            # Decay step size
            step *= decay_factor
            
            # If step is too small or we've been stagnant for too long, 
            # it's likely we've converged to a local minimum or are stuck.
            if step < min_step:
                break
                
            if stagnant_cycles > max_stagnant:
                # Escape local minimum by a random kick
                if eval_count < budget - 10:
                    kick = rng.uniform(-step, step, n)
                    x_kick = current_x + kick
                    x_kick = np.clip(x_kick, -5, 5)
                    f_kick = safe_func(x_kick)
                    if f_kick < current_f:
                        current_x = x_kick
                        current_f = f_kick
                    # Reset step size to a moderate value to explore again
                    step = 0.5
                    stagnant_cycles = 0
                else:
                    break

        # Phase 3: Final Polish
        # Use a very small step size to squeeze out the last bit of precision.
        # This is important for AOCC as it rewards early progress, but 
        # also the final value. However, AOCC is the mean of n_i, so 
        # getting close to f_opt early is better. 
        # Actually, AOCC = 1 - mean(n_i). n_i depends on b_i (best so far).
        # So we want b_i to drop quickly.
        # The CCD above drops b_i quickly.
        
        # If we have budget left, do a few more random restarts?
        # Lunacek's secondary minimum is a trap. 
        # If we haven't found the global minimum, a restart might help.
        # Heuristic: If current_f is not "very" low (relative to typical Lunacek values), try restart.
        # But we don't know f_opt. 
        # Let's just try one more restart from a random point if we have > 50 evals left.
        
        if eval_count < budget - 20:
            x_restart = rng.uniform(-4, 4, n)
            # Run a short CCD from here
            temp_x = x_restart.copy()
            temp_f = safe_func(temp_x)
            
            if temp_f < best_f:
                best_x = temp_x.copy()
                best_f = temp_f
                current_x = temp_x
                current_f = temp_f
                
                # Short CCD polish from new best
                step = 1.0
                for _ in range(5): # Max 5 cycles * 5 dims * 2 = 50 evals approx
                    if time.time() > deadline or eval_count >= budget - 5:
                        break
                    for i in range(n):
                        if time.time() > deadline or eval_count >= budget - 5:
                            break
                        x_p = current_x.copy()
                        x_p[i] += step
                        x_p = np.clip(x_p, -5, 5)
                        f_p = safe_func(x_p)
                        if f_p < current_f:
                            current_x = x_p
                            current_f = f_p
                        else:
                            x_m = current_x.copy()
                            x_m[i] -= step
                            x_m = np.clip(x_m, -5, 5)
                            f_m = safe_func(x_m)
                            if f_m < current_f:
                                current_x = x_m
                                current_f = f_m
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f

    def _cov_to_b_d(self, C, n):
        # Not used in CCD strategy, but kept for interface compatibility if needed
        try:
            L = np.linalg.cholesky(C)
            return L, np.ones(n)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs, np.sqrt(eigvals)
