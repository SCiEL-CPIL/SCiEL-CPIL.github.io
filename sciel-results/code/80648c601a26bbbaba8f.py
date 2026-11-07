"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB f14 (Different Powers).
    Strategy: Multi-start L-BFGS-B with immediate convergence attempts.
    Avoids wasting budget on pure evaluation; optimizes from diverse starts immediately.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, keep 10s buffer
        deadline = start_time + 320.0 
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        eval_count = 0

        def wrapped(x):
            nonlocal eval_count, best_x, best_fx
            if eval_count >= budget:
                return 1e10
            # Clip to box
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            eval_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        def time_ok(margin=2.0):
            return time.time() < deadline - margin

        # Generate diverse starting points
        # 1. Origin
        starts = [np.zeros(n)]
        
        # 2. Small perturbations around origin (to handle local minima near 0)
        for scale in [0.1, 0.5, 1.0]:
            for _ in range(3):
                starts.append(rng.normal(0, scale, n))
        
        # 3. Axis-aligned points (to handle shifts along axes)
        for d in range(n):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                pt = np.zeros(n)
                pt[d] = val
                starts.append(pt)
        
        # 4. Random points in the box
        for _ in range(15):
            starts.append(rng.uniform(-4.5, 4.5, n))
            
        # 5. Corner-like points (rotated)
        for _ in range(5):
            pt = rng.uniform(-4, 4, n)
            # Ensure some spread
            starts.append(pt)

        # Sort starts: Prefer points closer to origin (higher prior probability for many BBOB instances)
        # But keep the origin first.
        # Actually, for f14, the minimum is often not at the origin due to shifts.
        # However, starting at origin and letting L-BFGS-B move is usually very effective.
        # Sorting by norm is a safe heuristic.
        starts.sort(key=lambda x: np.linalg.norm(x))

        # Main Optimization Loop
        # We iterate through starts. For each, we run L-BFGS-B.
        # We allocate a portion of the remaining budget to each run.
        
        # Strategy for budget allocation per run:
        # - First run (origin): Give it a lot of iterations to converge deeply.
        # - Subsequent runs: Give them enough to find a local minimum, but not exhaust the budget if they fail.
        
        # Let's use a dynamic maxiter.
        # If we have 500 evals, and 25 starts, that's 20 evals/start.
        # L-BFGS-B in 5D might need 50-100 evals to converge well.
        # So we can't run 25 full convergences.
        # We will run from the most promising starts first.
        
        # We will run L-BFGS-B from each start, but cap the iterations 
        # such that we don't waste too much on a bad start.
        
        # However, if a start converges to the global min, we are done.
        # If it converges to a local min, we move on.
        
        # Let's try:
        # Run 1 (Origin): maxiter = min(remaining, 150)
        # Run 2-5: maxiter = min(remaining, 50)
        # Run 6+: maxiter = min(remaining, 20)
        
        run_idx = 0
        for x0 in starts:
            if not time_ok(5.0):
                break
            if eval_count >= budget:
                break
            
            remaining = budget - eval_count
            if remaining < 5:
                break
                
            # Determine maxiter for this run
            if run_idx == 0:
                maxiter = min(remaining, 150)
            elif run_idx < 5:
                maxiter = min(remaining, 50)
            else:
                maxiter = min(remaining, 20)
                
            if maxiter < 2:
                break

            try:
                res = minimize(
                    wrapped, x0, method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={'maxiter': maxiter, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # wrapped updates best_x/best_fx internally
            except Exception:
                pass
            
            run_idx += 1
            
            # Early stopping condition:
            # If we have found a very good solution (e.g., fx < 1e-6 relative to scale? 
            # We don't know f_opt. But if we converged tightly, maybe stop?)
            # Actually, let's just continue to ensure we don't miss a better basin.
            # But if we used most of the budget, the loop will break.

        # Final Polish
        # If we have budget left, polish the best point found with high precision
        if time_ok(2.0) and eval_count < budget and best_x is not None:
            remaining = budget - eval_count
            if remaining >= 5:
                try:
                    # Use tight tolerances
                    res = minimize(
                        wrapped, best_x, method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * n,
                        options={'maxiter': remaining, 'maxls': 30, 'ftol': 1e-15, 'gtol': 1e-10}
                    )
                except Exception:
                    pass
        
        # Fallback if no evaluation was made (should not happen)
        if best_x is None:
            best_x = np.zeros(n)
            if eval_count < budget:
                best_fx = wrapped(best_x)
            else:
                best_fx = 1e10

        return best_x, best_fx
# EVOLVE-BLOCK-END
