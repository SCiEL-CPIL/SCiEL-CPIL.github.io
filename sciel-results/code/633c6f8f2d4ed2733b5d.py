"""Multi-start Differential Evolution with adaptive (1+1)-ES polish for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

class Optimizer:
    """Multi-start DE with adaptive (1+1)-ES for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        
        n_calls = [0]
        stop_flag = [False]

        def safe_func(x):
            if stop_flag[0] or n_calls[0] >= budget or time.time() > deadline:
                stop_flag[0] = True
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except:
                return np.inf

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Multi-run Differential Evolution for global exploration
        # This configuration matches the high-performing parent (0.9789)
        # Run 1: Broad exploration (300 calls)
        # Run 2: Medium exploration (150 calls)
        # Run 3: Fine exploration (50 calls)
        runs = [300, 150, 50]
        
        for i, run_budget in enumerate(runs):
            if n_calls[0] >= budget or time.time() > deadline:
                break
            
            stop_flag[0] = False
            start_calls = n_calls[0]
            
            try:
                res = differential_evolution(
                    safe_func,
                    bounds,
                    maxiter=100,
                    popsize=15,
                    tol=1e-10,
                    seed=42 + i,
                    init='latinhypercube',
                    callback=lambda xk, convergence: True if (
                        stop_flag[0] or 
                        n_calls[0] - start_calls >= run_budget or 
                        time.time() > deadline
                    ) else False
                )
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Adaptive (1+1)-ES with 1/5-rule step-size adaptation
        # This is a structural change from Nelder-Mead, offering better robustness
        # for the non-smooth Weierstrass landscape in the final refinement phase.
        remaining = budget - n_calls[0]
        if remaining > 5 and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            
            # Initialize step size
            sigma = 0.1
            
            # 1/5-rule parameters
            rho = 1.0 / (5 * dim + 1)
            log_sigma = np.log(sigma)
            
            # RNG for reproducibility within the run
            rng = np.random.default_rng(12345)
            
            # Run (1+1)-ES until budget or time runs out
            while n_calls[0] < budget and time.time() < deadline and not stop_flag[0]:
                # Generate one offspring
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                
                # Evaluate
                f_trial = safe_func(x_trial)
                
                if not np.isfinite(f_trial):
                    break

                # Update best if improved
                if f_trial < f_x:
                    x = x_trial
                    f_x = f_trial
                    # Success: increase sigma
                    log_sigma += rho
                else:
                    # Failure: decrease sigma
                    log_sigma -= rho / 4.0
                
                sigma = np.exp(log_sigma)
                
                # Optional: If sigma becomes too small, we might stagnate, 
                # but for 50-100 calls, this is usually fine.
                # We can also break if sigma < 1e-10
                if sigma < 1e-10:
                    break

            # Final update
            if f_x < best_f:
                best_f = f_x
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
