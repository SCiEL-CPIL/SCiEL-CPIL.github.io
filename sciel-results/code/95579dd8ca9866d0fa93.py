"""Global-to-local (1+10)-ES optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global random search followed by (1+10)-ES with stagnation restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s wall-clock limit

        dim = self.dim
        total_budget = self.budget

        n_calls = 0
        best_f = float('inf')
        best_x = None

        rng = np.random.default_rng(42)

        def record(x, fx):
            nonlocal n_calls, best_f, best_x
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float, copy=True)

        # Phase 1: Extensive random exploration to identify the best basin
        # Use 50 uniform random evaluations to find a good starting point
        n_initial = 50
        for _ in range(n_initial):
            if n_calls >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            record(x, fx)

        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Phase 2: (1+10)-ES with restarts
        # lambda_pop = 10 offspring per generation
        lambda_pop = 10
        
        # Standard step size adaptation constants for (1+1) or (1+lambda)
        # Using a simple success-based adaptation or standard tau
        # tau = (1 + 0.5/sqrt(D)) * (1 + 0.5/D)
        tau = (1 + 0.5 / np.sqrt(dim)) * (1 + 0.5 / dim)
        
        # State for the current ES run
        x = best_x.copy()
        sigma = 1.0
        
        # Restart logic
        stagnation_counter = 0
        stagnation_limit = 20 # Restart if no improvement for 20 generations
        restart_count = 0
        max_restarts = 5 # Limit total restarts to avoid wasting budget on bad starts

        while n_calls < total_budget and time.time() < deadline:
            # Check if we can afford a full population
            if n_calls + lambda_pop > total_budget:
                # Use remaining budget for individual evaluations
                # Simple local search: try random steps with current sigma
                remaining = total_budget - n_calls
                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = func(y)
                    record(y, fy)
                    if fy < best_f:
                        x = y.copy()
                break

            # Generate population
            ys = []
            fs = []
            for j in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                record(y, fy)
                ys.append(y)
                fs.append(fy)

            ys = np.array(ys)
            fs = np.array(fs)
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # (1+lambda) selection: best offspring replaces parent if better
            if fs[0] < best_f:
                # Update global best
                best_f = fs[0]
                best_x = ys[0].copy()
                # Update local parent
                x = ys[0].copy()
                # Increase sigma slightly on success? Or keep standard tau
                # Standard (1+1) often uses 1/5th rule, but for (1+10) 
                # we can just use the standard sigma update if we were doing 
                # proper ES. Here we just accept the best and keep sigma 
                # constant or adapt it. 
                # Let's adapt sigma based on the distance moved? 
                # No, standard practice for simple ES: 
                # If we treat this as (1+1) with multiple trials, 
                # we might just keep sigma fixed or use a simple rule.
                # Let's use a simple rule: if success, sigma *= 1.1, else sigma *= 0.9
                # But we have 10 trials. Let's say if ANY trial improved, it's a success.
                sigma *= 1.1
                stagnation_counter = 0
            else:
                # No improvement in this generation
                sigma *= 0.9
                stagnation_counter += 1

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 10.0)

            # Check for restart
            if stagnation_counter > stagnation_limit:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                
                # Restart from global best
                x = best_x.copy()
                # Reset sigma to a moderate value to explore again
                sigma = 1.0
                stagnation_counter = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
