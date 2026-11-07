"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Sobol-based global search followed by (1+1)-ES local refinement.
    
    Structural change: Replaces the population-based Differential Evolution with a two-phase
    strategy. Phase 1 uses a deterministic Sobol sequence for systematic space-filling exploration
    to rapidly identify the global basin. Phase 2 uses a (1+1)-Evolution Strategy with adaptive
    step size and stagnation-based restarts for precise local refinement. This addresses the
    stochastic variance and potential stagnation of DE while maintaining strict budget adherence.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, global_best_x, global_best_fx
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline - 5.0

        # --- Phase 1: Sobol-based global exploration ---
        # Use a Sobol sequence to systematically sample the domain.
        # We allocate roughly 40% of the budget for global exploration.
        n_sobol = min(budget // 2, 200)
        
        # Generate Sobol sequence points in [0, 1]^dim
        # SciPy's qmc.Sobol is available, but we can also use a simple implementation or numpy if needed.
        # Let's use scipy.stats.qmc if available, otherwise fallback to random.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            # We need enough points. Sobol sequences are power-of-2 efficient.
            # Find the next power of 2 >= n_sobol
            n_points = 1
            while n_points < n_sobol:
                n_points *= 2
            sobol_points = sampler.random(n_points)
            # Scale to [-5, 5]
            sobol_points = sobol_points * 10 - 5
        except ImportError:
            # Fallback to random uniform if SciPy not available (though it is per contract)
            sobol_points = rng.uniform(-5, 5, size=(n_sobol, dim))

        for i in range(len(sobol_points)):
            if not time_ok() or calls_made >= budget:
                break
            x = sobol_points[i]
            fx = evaluate(x)
            if fx == np.inf:
                break

        if global_best_x is None:
            x0 = np.zeros(dim)
            evaluate(x0)

        # --- Phase 2: (1+1)-ES Local Refinement ---
        # Use the remaining budget for local refinement.
        # (1+1)-ES is efficient for smooth local landscapes and adapts step size.
        
        x_current = global_best_x.copy()
        fx_current = global_best_fx
        
        # Initial step size
        sigma = 1.0
        
        # Parameters for (1+1)-ES
        # Success rate controller or simple adaptation
        # We use a simple adaptation rule: increase sigma on success, decrease on failure.
        # Or use the 1/5th success rule.
        
        remaining_calls = budget - calls_made
        
        if remaining_calls > 0 and global_best_x is not None:
            # We will run (1+1)-ES until budget or time runs out.
            # To handle multimodality, we can restart if no improvement for a while.
            
            no_improve_count = 0
            max_no_improve = 50  # Restart after 50 steps without improvement
            
            while calls_made < budget and time_ok():
                # Generate mutant
                y = x_current + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = evaluate(y)
                
                if fy == np.inf:
                    break
                
                if fy < fx_current:
                    # Success
                    x_current = y
                    fx_current = fy
                    no_improve_count = 0
                    # Increase step size slightly (or keep same, depending on strategy)
                    # For (1+1)-ES, a common rule is to increase sigma by a factor if success
                    sigma *= 1.1
                else:
                    # Failure
                    no_improve_count += 1
                    # Decrease step size
                    sigma *= 0.9
                    
                # Stagnation check
                if no_improve_count >= max_no_improve:
                    # Restart from a new random point near the best found so far?
                    # Or just reset sigma and continue?
                    # Let's try a local restart: perturb the best point significantly
                    if calls_made < budget - 10:
                        x_current = global_best_x + 2.0 * rng.standard_normal(dim)
                        x_current = np.clip(x_current, -5, 5)
                        fx_current = evaluate(x_current)
                        if fx_current == np.inf:
                            break
                        sigma = 1.0
                        no_improve_count = 0
                    else:
                        break

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
