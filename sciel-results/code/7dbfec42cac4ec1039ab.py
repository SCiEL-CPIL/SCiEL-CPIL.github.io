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

        # Phase 2: Multi-start L-BFGS-B polishing
        # L-BFGS-B exploits local curvature to take large directed steps along the Rosenbrock valley.
        # We try multiple starting points from the DE population to ensure we find the best basin.
        
        if calls_made < budget and time.time() < deadline - 10.0:
            # Collect candidate starting points: best_x plus a few diverse points from DE
            candidates = [best_x.copy()]
            
            # Add some diverse points: perturb best_x in different directions
            for scale in [0.5, 1.0, 2.0]:
                for _ in range(3):
                    perturbation = np.random.uniform(-scale, scale, size=dim)
                    x_cand = best_x + perturbation
                    x_cand = np.clip(x_cand, -5.0, 5.0)
                    candidates.append(x_cand)
            
            # Sort candidates by function value (evaluate them first)
            candidate_scores = []
            for x_cand in candidates:
                if calls_made >= budget or time.time() > deadline - 5.0:
                    break
                f_cand = wrapper(x_cand)
                candidate_scores.append((f_cand, x_cand.copy()))
            
            # Sort by function value, keep top 5
            candidate_scores.sort(key=lambda t: t[0])
            top_candidates = [x for _, x in candidate_scores[:5]]
            
            # Run L-BFGS-B from each top candidate
            for x_start in top_candidates:
                if calls_made >= budget or time.time() > deadline - 2.0:
                    break
                
                try:
                    res_lbfgsb = minimize(
                        wrapper,
                        x_start,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-12}
                    )
                    # Wrapper updates best_x/best_f automatically
                except Exception:
                    pass
        
        # Phase 3: Final fine-grained local search if budget remains
        if calls_made < budget and time.time() < deadline - 2.0:
            # Try a few random steps around the best point with decreasing step sizes
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
            for step_size in step_sizes:
                if calls_made >= budget or time.time() > deadline:
                    break
                perturbation = np.random.uniform(-step_size, step_size, size=dim)
                x_new = best_x + perturbation
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = wrapper(x_new)
                if f_new < best_f:
                    # Improved, continue with smaller steps
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
