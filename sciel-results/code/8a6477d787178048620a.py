"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and coordinate-descent refinement for BBOB-style task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _coordinate_descent(self, func, x, f, budget, dim, deadline):
        """Perform coordinate descent refinement. Returns (best_x, best_f, evals_used)."""
        best_x = x.copy()
        best_f = f
        evals = 0
        
        # Adaptive step sizes per coordinate
        step_sizes = np.ones(dim) * 0.5
        min_step = 1e-8
        max_step = 5.0
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            improved = False
            for d in range(dim):
                if evals >= budget or time.time() > deadline:
                    break
                
                # Try positive step
                trial = best_x.copy()
                trial[d] = np.clip(trial[d] + step_sizes[d], -5, 5)
                try:
                    ft = func(trial)
                except Exception:
                    ft = np.inf
                evals += 1
                
                if ft < best_f:
                    best_x = trial
                    best_f = ft
                    improved = True
                else:
                    # Try negative step
                    trial = best_x.copy()
                    trial[d] = np.clip(trial[d] - step_sizes[d], -5, 5)
                    try:
                        ft = func(trial)
                    except Exception:
                        ft = np.inf
                    evals += 1
                    
                    if ft < best_f:
                        best_x = trial
                        best_f = ft
                        improved = True
                    else:
                        # No improvement in this coordinate, reduce step size
                        step_sizes[d] *= 0.5
                        if step_sizes[d] < min_step:
                            step_sizes[d] = min_step
            
            # If no coordinate improved, we've likely converged
            if not improved:
                break
            
            # Occasionally try to increase step sizes if we're making good progress
            # This helps escape local minima in coordinate space
            if evals % 100 < dim:
                step_sizes = np.clip(step_sizes * 1.2, min_step, max_step)
        
        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Phase 1: (1+1)-ES with 1/5 success rule
        sigma = 3.0  # Initial step size to cover search space
        x = np.zeros(dim)
        
        try:
            f = func(x)
        except Exception:
            f = np.inf
        
        best_x = x.copy()
        best_f = f
        evals = 1
        
        # 1/5 success rule parameters
        window = 10
        successes = 0
        trial_count = 0
        
        while evals < budget and time.time() < deadline:
            # Generate candidate
            direction = rng.standard_normal(dim)
            candidate = x + sigma * direction
            np.clip(candidate, -5, 5, out=candidate)
            
            try:
                f_candidate = func(candidate)
            except Exception:
                f_candidate = np.inf
            
            evals += 1
            trial_count += 1
            
            if f_candidate < f:
                x = candidate
                f = f_candidate
                successes += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            # Update sigma using 1/5 success rule
            if trial_count >= window:
                success_rate = successes / trial_count
                # Exponential update for smooth adaptation
                sigma *= np.exp(0.5 * (success_rate - 0.2))
                sigma = np.clip(sigma, 1e-8, 20.0)
                successes = 0
                trial_count = 0
            
            # If sigma becomes very small, we've likely converged
            if sigma < 1e-6:
                break
        
        # Phase 2: Coordinate descent refinement around best point
        if evals < budget and time.time() < deadline:
            remaining_budget = budget - evals
            # Use a portion of remaining budget for coordinate descent
            cd_budget = min(remaining_budget, max(50, remaining_budget // 2))
            
            try:
                refined_x, refined_f, cd_evals = self._coordinate_descent(
                    func, best_x, best_f, cd_budget, dim, deadline
                )
                evals += cd_evals
                
                if refined_f < best_f:
                    best_f = refined_f
                    best_x = refined_x
            except Exception:
                pass
        
        # Phase 3: If we still have budget, do another round of (1+1)-ES with smaller sigma
        if evals < budget and time.time() < deadline:
            sigma = 0.5  # Smaller step size for fine-tuning
            x = best_x.copy()
            f = best_f
            
            successes = 0
            trial_count = 0
            
            while evals < budget and time.time() < deadline:
                direction = rng.standard_normal(dim)
                candidate = x + sigma * direction
                np.clip(candidate, -5, 5, out=candidate)
                
                try:
                    f_candidate = func(candidate)
                except Exception:
                    f_candidate = np.inf
                
                evals += 1
                trial_count += 1
                
                if f_candidate < f:
                    x = candidate
                    f = f_candidate
                    successes += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                
                if trial_count >= window:
                    success_rate = successes / trial_count
                    sigma *= np.exp(0.5 * (success_rate - 0.2))
                    sigma = np.clip(sigma, 1e-10, 5.0)
                    successes = 0
                    trial_count = 0
                
                if sigma < 1e-8:
                    break
        
        # Fallback if nothing was found
        if best_x is None or np.isinf(best_f):
            x = rng.uniform(-5, 5, dim)
            try:
                f = func(x)
            except Exception:
                f = np.inf
            best_x = x
            best_f = f
        
        return best_x, best_f
# EVOLVE-BLOCK-END
