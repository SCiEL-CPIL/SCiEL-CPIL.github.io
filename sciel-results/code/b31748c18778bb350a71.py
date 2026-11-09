"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Portfolio optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Strategy:
      - Random sampling phase to get several diverse starting points.
      - Run multiple (1+1)-ES runs from the best samples, each with a
        sub-budget and adaptive step size (1/5 success rule variant).
      - Keep a global best across all runs.
      - Explicit wall-clock deadline guard (safety margin before 330 s).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # safety margin before 330s limit

        def safe_call(x):
            """Call func with time check, return (success, value)."""
            if time.time() > deadline:
                return False, None
            try:
                val = func(np.asarray(x, dtype=float))
                return True, float(val)
            except Exception:
                return False, None

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find diverse starting points
        n_samples = min(30, budget // 10)
        samples = []
        for _ in range(n_samples):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            ok, fx = safe_call(x)
            if ok:
                samples.append((fx, x.copy()))

        if not samples:
            # Fallback: just return the last valid point
            return np.zeros(dim), 0.0

        # Sort by fitness and pick top candidates
        samples.sort(key=lambda t: t[0])
        
        # Select top K starting points (avoid too many to leave budget for ES)
        k_starts = min(5, len(samples))
        start_points = [samples[i][1] for i in range(k_starts)]
        global_best_x = samples[0][1].copy()
        global_best_f = samples[0][0]

        # Phase 2: Run (1+1)-ES from each starting point
        # Distribute remaining budget among runs
        remaining_budget = budget - len(samples)
        if remaining_budget <= 0:
            return global_best_x, global_best_f

        # Allocate budget to each run, with more budget for better starts
        # Use a simple allocation: give each run an equal share, but prioritize
        # the best start
        n_runs = k_starts
        budget_per_run = max(10, remaining_budget // n_runs)
        
        # Run ES from each start point
        for run_idx in range(n_runs):
            if time.time() > deadline:
                break
            
            x = start_points[run_idx].copy()
            fx = func(x)  # We already have this, but re-evaluate to be safe
            # Actually, we have fx from samples, but let's use the stored value
            # to save budget. However, since we're in a new run, let's just use
            # the stored value as the initial evaluation.
            
            # Re-initialize sigma based on run
            sigma = 1.0
            success_count = 0
            total_steps = 0
            
            # Adaptive (1+1)-ES with 1/5 rule
            # We'll use a fixed number of steps per run, or until budget is used
            max_steps = budget_per_run
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                ok, fy = safe_call(y)
                if not ok:
                    break
                
                total_steps += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    
                    # Update global best
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                    
                    # Increase sigma on success
                    sigma *= 1.3
                else:
                    # Decrease sigma on failure
                    sigma *= 0.7
                
                # 1/5 rule: adjust sigma based on success rate every 10 steps
                if total_steps % 10 == 0:
                    success_rate = success_count / 10
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.9
                    success_count = 0
                
                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            # After each run, if we have budget left, consider restarting
            # from the global best with a smaller sigma for local refinement
            if time.time() > deadline:
                break
            
            # Local refinement from global best
            if budget_per_run > 20:
                x_refine = global_best_x.copy()
                fx_refine = global_best_f
                sigma_refine = 0.1
                refine_steps = min(20, budget_per_run // 4)
                
                for _ in range(refine_steps):
                    if time.time() > deadline:
                        break
                    
                    y = np.clip(x_refine + sigma_refine * rng.standard_normal(dim), -5, 5)
                    ok, fy = safe_call(y)
                    if not ok:
                        break
                    
                    if fy < fx_refine:
                        x_refine = y
                        fx_refine = fy
                        if fx_refine < global_best_f:
                            global_best_f = fx_refine
                            global_best_x = x_refine.copy()
                        sigma_refine *= 1.2
                    else:
                        sigma_refine *= 0.8
                    
                    sigma_refine = np.clip(sigma_refine, 1e-8, 1.0)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
