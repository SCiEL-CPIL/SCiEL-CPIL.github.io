"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Coordinate-descent hybrid with adaptive step sizes and random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        evals = 0
        
        def time_exceeded():
            return time.time() > deadline
        
        def eval_func(x):
            nonlocal evals, best_f, best_x
            x = np.clip(x, -5, 5)
            try:
                f_val = func(x)
            except Exception:
                f_val = np.inf
            evals += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global exploration via random restarts with coordinate descent
        # We allocate ~60% of budget to this phase
        phase1_budget = int(budget * 0.6)
        
        # Number of restarts: use remaining budget wisely
        # Each restart: initial point + coordinate descent iterations
        # Coordinate descent: dim dimensions * 2 directions * ~5 line search steps
        # So one restart costs roughly 1 + dim * 2 * 5 = 51 evals for dim=5
        # Let's say we can afford about phase1_budget / 30 restarts on average
        
        num_restarts = max(1, int(phase1_budget / 40))
        evals_per_restart = max(10, phase1_budget // num_restarts)
        
        for r in range(num_restarts):
            if evals >= phase1_budget or time_exceeded():
                break
            
            # Generate initial point for this restart
            # Mix of center, random, and previous best perturbation
            if r == 0:
                x = np.zeros(dim)
            elif r == 1:
                x = rng.uniform(-3, 3, size=dim)
            else:
                # Perturb previous best
                if np.isfinite(best_f):
                    x = best_x + rng.normal(0, 1.0, size=dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-4, 4, size=dim)
            
            fx = eval_func(x)
            if not np.isfinite(fx):
                continue
            
            # Adaptive coordinate descent with line search
            # Step sizes per dimension
            step_sizes = np.ones(dim) * 0.5
            
            # Number of CD iterations
            max_cd_iters = 3
            for it in range(max_cd_iters):
                if evals >= phase1_budget or time_exceeded():
                    break
                
                improved = False
                for d in range(dim):
                    if evals >= phase1_budget or time_exceeded():
                        break
                    
                    # Line search along dimension d
                    # Try several step sizes
                    best_trial_f = fx
                    best_trial_x = x.copy()
                    
                    # Adaptive step: try current step, then larger, then smaller
                    step_candidates = [step_sizes[d], step_sizes[d] * 2, step_sizes[d] * 0.5,
                                       step_sizes[d] * 4, step_sizes[d] * 0.25]
                    
                    for s in step_candidates:
                        if evals >= phase1_budget or time_exceeded():
                            break
                        for direction in [-1, 1]:
                            if evals >= phase1_budget or time_exceeded():
                                break
                            trial = x.copy()
                            trial[d] = x[d] + direction * s
                            trial = np.clip(trial, -5, 5)
                            ft = eval_func(trial)
                            if ft < best_trial_f:
                                best_trial_f = ft
                                best_trial_x = trial.copy()
                    
                    if best_trial_f < fx - 1e-10:
                        x = best_trial_x
                        fx = best_trial_f
                        improved = True
                        # Adapt step size: keep it if it worked
                        step_sizes[d] = min(step_sizes[d] * 1.5, 2.0)
                    else:
                        # No improvement: shrink step size
                        step_sizes[d] = max(step_sizes[d] * 0.5, 0.01)
                
                if not improved:
                    break
        
        # Phase 2: Local refinement around the best point found
        if evals < budget and not time_exceeded():
            x = best_x.copy()
            fx = best_f
            step_sizes = np.ones(dim) * 0.1
            
            # Fine-grained coordinate descent
            while evals < budget and not time_exceeded():
                improved = False
                for d in range(dim):
                    if evals >= budget or time_exceeded():
                        break
                    
                    best_trial_f = fx
                    best_trial_x = x.copy()
                    
                    # Try multiple step sizes for fine-tuning
                    for s in [step_sizes[d], step_sizes[d] * 0.5, step_sizes[d] * 2,
                              step_sizes[d] * 0.1, step_sizes[d] * 0.2]:
                        if evals >= budget or time_exceeded():
                            break
                        for direction in [-1, 1]:
                            if evals >= budget or time_exceeded():
                                break
                            trial = x.copy()
                            trial[d] = x[d] + direction * s
                            trial = np.clip(trial, -5, 5)
                            ft = eval_func(trial)
                            if ft < best_trial_f:
                                best_trial_f = ft
                                best_trial_x = trial.copy()
                    
                    if best_trial_f < fx - 1e-10:
                        x = best_trial_x
                        fx = best_trial_f
                        improved = True
                        step_sizes[d] = min(step_sizes[d] * 1.2, 0.5)
                    else:
                        step_sizes[d] = max(step_sizes[d] * 0.7, 0.001)
                
                if not improved:
                    break
                
                # Occasionally add a small random perturbation to escape local minima
                if evals < budget - 10 and not time_exceeded() and rng.random() < 0.1:
                    perturbation = rng.normal(0, 0.05, size=dim)
                    trial = x + perturbation
                    trial = np.clip(trial, -5, 5)
                    ft = eval_func(trial)
                    if ft < fx:
                        x, fx = trial, ft

        return best_x, best_f
# EVOLVE-BLOCK-END
