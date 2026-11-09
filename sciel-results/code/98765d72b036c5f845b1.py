"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

class Optimizer:
    """
    (1+1)-ES based optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Strategy:
      - Use (1+1)-ES with 1/5 success rule for step size adaptation.
      - Multi-restart policy: restart if no improvement for a while or if stuck.
      - Fast and robust, suitable for 5D with 500 function calls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        dim = self.dim
        n_evals = 0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')

        # We will do multiple restarts.
        # Each restart gets a fraction of the budget, or we just loop until budget is done.
        # Given 500 calls, we can do many restarts.
        # Let's do a "budgeted restart" approach:
        # Each run uses up to 50 calls, or until no improvement for 20 calls.
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break

            # Initialize current run
            # Start near the global best if we have one, otherwise random
            if best_x is not None:
                x = best_x + rng.standard_normal(dim) * 0.5
            else:
                x = rng.uniform(-5, 5, dim)
            
            # Clip to box
            x = np.clip(x, -5, 5)
            
            f = func(x)
            n_evals += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            sigma = 2.0  # Initial step size for this run
            no_improve_count = 0
            max_no_improve = 20
            run_budget = 50  # Max calls per restart
            
            for _ in range(run_budget):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Generate mutant
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                n_evals += 1
                
                if fy < f:
                    # Success
                    x = y
                    f = fy
                    no_improve_count = 0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    # Increase sigma slightly (1/5 rule: increase if success rate > 1/5)
                    # For (1+1), we usually do:
                    # sigma = sigma * (1 + 1/5) if success else sigma * (1 - 1/5)
                    # But a common robust version is:
                    sigma *= 1.1
                else:
                    # Failure
                    no_improve_count += 1
                    sigma *= 0.9
                
                # Early stop this restart if stuck
                if no_improve_count > max_no_improve:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
