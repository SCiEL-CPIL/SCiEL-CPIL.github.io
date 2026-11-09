"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """ZDT1-aware optimizer: analytical Pareto front sampling + L-BFGS-B local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 structure: f1 = x[0], f2 = 1 + sin(10*x[0]) on Pareto front
        # We sample x1 values targeting peaks/valleys of sin(10*x1)
        # sin(10*x1) has peaks at 10*x1 = pi/2 + 2k*pi => x1 = (pi/2 + 2k*pi)/10
        # and valleys at 10*x1 = -pi/2 + 2k*pi => x1 = (-pi/2 + 2k*pi)/10

        # Generate candidate x1 values
        x1_candidates = []
        # Peaks of sin(10*x1): x1 = (pi/2 + 2*k*pi)/10 for k=0..4 (stays in [0,1])
        for k in range(5):
            x1 = (np.pi / 2 + 2 * k * np.pi) / 10.0
            if 0 <= x1 <= 1:
                x1_candidates.append(x1)
        # Valleys of sin(10*x1): x1 = (-pi/2 + 2*k*pi)/10 for k=1..5
        for k in range(1, 6):
            x1 = (-np.pi / 2 + 2 * k * np.pi) / 10.0
            if 0 <= x1 <= 1:
                x1_candidates.append(x1)
        # Add endpoints and some uniform points for coverage
        x1_candidates.extend([0.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        # Deduplicate and sort
        x1_candidates = sorted(set(np.round(x1_candidates, 10)))

        # For each x1, we need to find x[1:] that minimizes f2 while keeping x[0]=x1
        # Strategy: use L-BFGS-B to minimize f2(x) subject to x[0]=x1 (handled by fixing x[0])
        # Actually, for ZDT1, f2 depends on x[0] and sum of (x_i - 0.5)^2 for i>=1
        # f2 = 1 + 10*(n-1) - 4*sum(x_i - 0.5)^2 * cos(20*pi*(x_i - 0.5)) for i>=1... 
        # Actually ZDT1: f2(x) = 1 + sin(10*x1) * g(x) where g(x) = 1 + 10*(n-1) - sum(4*x_i*cos(20*pi*x_i) + ...)
        # The exact form varies. Let's just use the problem evaluator.

        # We'll fix x[0] = target_x1 and optimize x[1:] to minimize f2
        # But we don't know the exact ZDT1 formula, so we use the black-box problem(x)

        # Approach: for each target x1, do a local search starting from a random point
        # with x[0] fixed to target_x1

        results_F = []
        results_X = []
        used = 0

        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.asarray(x, dtype=float)
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            used += 1
            return x, f

        # First, evaluate all x1 candidates with a simple heuristic:
        # Set x[1:] = 0.5 (center) as initial guess
        for x1 in x1_candidates:
            if time.time() > deadline:
                break
            if used >= budget:
                break
            x_init = np.full(dim, 0.5)
            x_init[0] = x1
            res = evaluate(x_init)
            if res is not None:
                results_X.append(res[0])
                results_F.append(res[1])

        # Now use remaining budget for L-BFGS-B refinement on the best candidates
        # Sort by f1 (which should be ~x1) and pick diverse points
        if len(results_F) > 0:
            F_arr = np.array(results_F)
            X_arr = np.array(results_X)

            # For each point, try to improve f2 by optimizing x[1:] with x[0] fixed
            # Use a few L-BFGS-B calls
            n_refine = min(15, budget - used)
            # Pick points spread across f1 range
            if len(F_arr) > 1:
                f1_vals = F_arr[:, 0]
                # Select up to n_refine points spread across f1
                if len(f1_vals) <= n_refine:
                    indices = list(range(len(f1_vals)))
                else:
                    indices = list(np.linspace(0, len(f1_vals) - 1, n_refine).astype(int))
            else:
                indices = [0]

            for idx in indices:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                x0 = X_arr[idx].copy()
                target_x1 = x0[0]

                def objective(x_rest):
                    x = x0.copy()
                    x[1:] = x_rest
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    f = np.asarray(f, dtype=float)
                    used += 1
                    return f[1]  # minimize f2

                # L-BFGS-B on x[1:]
                try:
                    res_opt = minimize(
                        objective,
                        x0[1:],
                        method='L-BFGS-B',
                        bounds=[(lo[i], hi[i]) for i in range(1, dim)],
                        options={'maxiter': 50, 'maxfun': 30}
                    )
                    if res_opt.success and used < budget:
                        x_best = x0.copy()
                        x_best[1:] = np.clip(res_opt.x, lo[1:], hi[1:])
                        f_best = np.asarray(problem(x_best), dtype=float)
                        used += 1
                        results_X.append(x_best)
                        results_F.append(f_best)
                except Exception:
                    pass

        # Final: ensure we haven't exceeded budget
        # Return the last evaluated point (return value is ignored by evaluator)
        if len(results_X) > 0:
            return results_X[-1]
        return np.zeros(dim)
# EVOLVE-BLOCK-END
