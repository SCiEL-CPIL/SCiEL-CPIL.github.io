"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """
    Hybrid optimizer for BBOB f9 (rotated Rosenbrock).
    Combines Differential Evolution for global exploration with L-BFGS-B for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 300s to be safe.
        deadline = start_time + 280.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            # Time check
            if time.time() > deadline:
                return best_f
            # Budget check
            if calls_made >= budget:
                return best_f

            x = np.clip(x, -5.0, 5.0)
            calls_made += 1

            try:
                f_val = func(x)
            except Exception:
                return best_f

            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Search using Differential Evolution
        # DE is robust to multimodality and good for finding the basin of attraction.
        # We allocate ~60% of the budget to this phase.
        de_budget = int(budget * 0.6)
        
        # DE parameters:
        # popsize: 15 (good for 5D)
        # maxiter: estimated to fit de_budget
        # Roughly popsize * maxiter calls.
        # 15 * 20 = 300 calls. If budget is 500, de_budget is 300. Perfect.
        maxiter_de = max(1, de_budget // 15)
        
        try:
            res_de = differential_evolution(
                wrapper,
                bounds=[(-5.0, 5.0)] * dim,
                popsize=15,
                maxiter=maxiter_de,
                seed=42,
                tol=1e-8,
                init='latinhypercube',
                polish=False,  # We will do our own polishing
                update_interval=1
            )
        except Exception:
            pass

        if best_x is None:
            # Fallback if DE failed completely
            best_x = np.zeros(dim)
            best_f = wrapper(best_x)

        # Phase 2: Local Polish using L-BFGS-B
        # L-BFGS-B is very efficient for smooth functions like Rosenbrock once in the valley.
        # It uses gradient approximations (finite differences), costing ~dim+1 calls per step.
        # Remaining budget: budget - calls_made
        remaining_budget = budget - calls_made
        
        if remaining_budget > 0 and time.time() < deadline - 5.0:
            try:
                # maxiter for L-BFGS-B should be generous, but budget is the hard limit
                # via the wrapper.
                res_lbfgsb = minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 1000, 'gtol': 1e-15, 'ftol': 1e-15}
                )
                # Wrapper updates best_x/best_f automatically
            except Exception:
                pass

        # Phase 3: Final Nelder-Mead or Simple Local Search if budget remains
        # Sometimes L-BFGS-B stops early due to gtol but can still improve with small steps.
        # Or use a few random perturbations around best_x.
        if calls_made < budget and time.time() < deadline - 2.0:
            # Try a few random steps around the best point
            for _ in range(10):
                if calls_made >= budget or time.time() > deadline:
                    break
                step_size = 0.1
                perturbation = np.random.uniform(-step_size, step_size, size=dim)
                x_new = best_x + perturbation
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = wrapper(x_new)
                if f_new < best_f:
                    # If improved, maybe do a quick L-BFGS-B restart?
                    # For simplicity, just continue random perturbations or break to save time
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
