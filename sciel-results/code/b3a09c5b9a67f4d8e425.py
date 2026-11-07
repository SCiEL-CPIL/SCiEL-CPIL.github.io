"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart optimizer with two-phase search for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        dim = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        # We will use multiple restarts. 
        # Let's try 5 restarts, each with budget/5 evaluations.
        # This allows escaping local minima which are common in f6.
        n_restarts = 5
        budget_per_restart = total_budget // n_restarts
        
        # Ensure we have at least 10 evals per restart if possible
        if budget_per_restart < 10:
            n_restarts = max(1, total_budget // 10)
            budget_per_restart = total_budget // n_restarts

        global_best_x = None
        global_best_f = np.inf
        calls_made = 0

        # Pre-generate a set of diverse starting points
        # Mix of center, corners, and random
        start_points = []
        # 1. Center
        start_points.append(np.zeros(dim))
        # 2. Random points
        for _ in range(n_restarts - 1):
            start_points.append(rng.uniform(-4, 4, dim))
        
        # Shuffle start points to ensure diversity across restarts
        rng.shuffle(start_points)

        for r in range(n_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            # Pick a start point
            if r < len(start_points):
                x = start_points[r].copy()
            else:
                x = rng.uniform(-4, 4, dim)
            
            fx = func(x)
            calls_made += 1
            
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

            # Phase 1: Coarse (1+1)-ES for global exploration
            sigma = 2.0
            successes = 0
            attempts = 0
            local_best_x = x.copy()
            local_best_f = fx
            local_calls = 1
            
            # Run until budget for this restart is exhausted or time limit
            while calls_made < total_budget and local_calls < budget_per_restart:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                local_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                else:
                    pass

                attempts += 1

                # 1/5 success rule
                if attempts >= 5:
                    success_rate = successes / attempts
                    if success_rate > 0.2:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    successes = 0
                    attempts = 0
                    sigma = max(0.01, min(sigma, 10.0))

            # Phase 2: Fine polishing using coordinate descent / small perturbations
            # Use remaining budget from this restart or a fixed small amount if budget allows
            # Actually, let's just use a few specific small steps along axes from the local best
            # This exploits the axis-aligned features mentioned in inspirations.
            
            # We have spent local_calls out of budget_per_restart.
            # Let's reserve the last 10% of the restart budget for polishing.
            # Or simply do a few coordinate sweeps if we have time and budget.
            
            # Let's do a simple coordinate-wise refinement on the local best
            # We will use up to 5 evaluations per dimension (dim*5) for polishing
            # But we must respect total budget.
            
            polish_budget = min(5 * dim, max(0, budget_per_restart - local_calls))
            
            if polish_budget > 0:
                px = local_best_x.copy()
                pf = local_best_f
                
                # Try small steps along each axis
                step_sizes = [0.5, 0.1, 0.01]
                for step in step_sizes:
                    for d in range(dim):
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        if polish_budget <= 0:
                            break
                        
                        # Positive step
                        y_plus = px.copy()
                        y_plus[d] = np.clip(y_plus[d] + step, -5, 5)
                        fy_plus = func(y_plus)
                        calls_made += 1
                        polish_budget -= 1
                        if fy_plus < pf:
                            px, pf = y_plus, fy_plus
                            if pf < global_best_f:
                                global_best_f = pf
                                global_best_x = px.copy()
                            
                        # Negative step
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        if polish_budget <= 0:
                            break
                            
                        y_minus = px.copy()
                        y_minus[d] = np.clip(y_minus[d] - step, -5, 5)
                        fy_minus = func(y_minus)
                        calls_made += 1
                        polish_budget -= 1
                        if fy_minus < pf:
                            px, pf = y_minus, fy_minus
                            if pf < global_best_f:
                                global_best_f = pf
                                global_best_x = px.copy()

        # If we ran out of budget or time, return the best found
        if global_best_x is None:
            # Fallback: should not happen if budget > 0
            x = np.zeros(dim)
            global_best_f = func(x)
            global_best_x = x
            calls_made += 1

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
