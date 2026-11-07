"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Coordinate-descent hybrid with adaptive per-dim steps for BBOB Schwefel."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Initialize at center
        x = np.zeros(dim)
        try:
            f_x = func(x)
        except Exception:
            f_x = np.inf
        evals = 1
        
        best_x = x.copy()
        best_f = f_x
        
        # Per-dimension step sizes
        steps = np.ones(dim) * 0.5
        
        # Phase 1: Coordinate descent with adaptive steps
        # Alternate between deterministic line search and random perturbations
        cd_budget = budget // 2  # Use half budget for coordinate descent
        perturbation_budget = budget - cd_budget
        
        # Coordinate descent phase
        while evals < cd_budget and time.time() < deadline:
            improved = False
            for d in range(dim):
                if evals >= cd_budget or time.time() >= deadline:
                    break
                
                # Try positive step
                x_trial = x.copy()
                x_trial[d] += steps[d]
                np.clip(x_trial, -5, 5, out=x_trial)
                
                try:
                    f_trial = func(x_trial)
                except Exception:
                    f_trial = np.inf
                evals += 1
                
                if f_trial < f_x:
                    x = x_trial
                    f_x = f_trial
                    improved = True
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                    # Increase step if successful
                    steps[d] = min(steps[d] * 1.2, 2.0)
                else:
                    # Try negative step
                    x_trial = x.copy()
                    x_trial[d] -= steps[d]
                    np.clip(x_trial, -5, 5, out=x_trial)
                    
                    try:
                        f_trial = func(x_trial)
                    except Exception:
                        f_trial = np.inf
                    evals += 1
                    
                    if f_trial < f_x:
                        x = x_trial
                        f_x = f_trial
                        improved = True
                        if f_x < best_f:
                            best_f = f_x
                            best_x = x.copy()
                        # Increase step if successful
                        steps[d] = min(steps[d] * 1.2, 2.0)
                    else:
                        # Decrease step if neither direction improved
                        steps[d] = max(steps[d] * 0.7, 0.01)
            
            # If no improvement in full cycle, add small random perturbation
            if not improved:
                # Random perturbation to escape local minima
                noise = rng.standard_normal(dim) * 0.1
                x_trial = x + noise
                np.clip(x_trial, -5, 5, out=x_trial)
                
                try:
                    f_trial = func(x_trial)
                except Exception:
                    f_trial = np.inf
                evals += 1
                
                if f_trial < f_x:
                    x = x_trial
                    f_x = f_trial
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
        
        # Phase 2: Random perturbation phase to explore further
        perturbation_sigma = 0.5
        while evals < budget and time.time() < deadline:
            # Generate random perturbation
            z = rng.standard_normal(dim)
            x_trial = x + perturbation_sigma * z
            np.clip(x_trial, -5, 5, out=x_trial)
            
            try:
                f_trial = func(x_trial)
            except Exception:
                f_trial = np.inf
            evals += 1
            
            if f_trial < f_x:
                x = x_trial
                f_x = f_trial
                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
                # Increase perturbation size slightly
                perturbation_sigma = min(perturbation_sigma * 1.1, 3.0)
            else:
                # Decrease perturbation size
                perturbation_sigma = max(perturbation_sigma * 0.9, 0.05)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
