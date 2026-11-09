"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Constrained multi-objective search using Augmented Epsilon-Constraint Method."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # We will use L-BFGS-B with numerical gradients (approx_grad=True).
        # Each call to problem(x) is one evaluation. We need to count carefully.
        
        # Strategy:
        # 1. Do a small random search to get an initial idea of the objective ranges.
        # 2. Use the Epsilon-Constraint method: minimize f0 subject to f1 <= epsilon.
        #    We sweep epsilon from f1_max to f1_min.
        # 3. We need to be careful with the budget. L-BFGS-B uses many evaluations per run.
        #    Let's estimate: L-BFGS-B with numerical gradients uses about 2*dim evaluations per iteration.
        #    Typical iterations: 20-50. So ~40-200 evals per run.
        #    With budget 200, we can afford 1-5 runs.
        
        # Let's be more strategic:
        # - Use 20 evaluations for initial random sampling to estimate f0_min, f0_max, f1_min, f1_max.
        # - Then use the remaining 180 evaluations for 3-4 L-BFGS-B runs at different epsilon values.
        
        # Actually, let's think about this differently. The problem is RE21, which has 4 variables.
        # The objectives are likely smooth. L-BFGS-B should converge quickly.
        
        # Let's do:
        # - 10 random points to get bounds on objectives.
        # - Then 4 L-BFGS-B runs, each with a budget of ~45 evaluations.
        
        # But we need to count evaluations precisely. Let's wrap problem to count.
        
        eval_count = [0]
        
        def count_problem(x):
            eval_count[0] += 1
            x = np.clip(x, lo, hi)
            return problem(x)
        
        # Phase 1: Random sampling to estimate objective ranges
        n_init = 10
        rng = np.random.default_rng(0)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([count_problem(x) for x in X_init])
        
        f0_min, f0_max = F_init[:, 0].min(), F_init[:, 0].max()
        f1_min, f1_max = F_init[:, 1].min(), F_init[:, 1].max()
        
        # Phase 2: Epsilon-Constraint sweep
        # We'll pick 4 epsilon values: f1_max, (f1_max+f1_min)/2, f1_min, and a bit below f1_min
        # Actually, let's use: f1_max, f1_max*0.8, f1_max*0.6, f1_max*0.4, f1_min
        # But we need to be careful about the scale.
        
        # Let's use linearly spaced epsilon values
        n_eps = 4
        epsilons = np.linspace(f1_max, f1_min, n_eps)
        
        # For each epsilon, minimize f0 subject to f1 <= epsilon
        # Using L-BFGS-B with a penalty for constraint violation
        
        all_points = list(X_init)
        all_objectives = list(F_init)
        
        remaining_budget = self.budget - eval_count[0]
        
        if remaining_budget <= 0:
            return np.array(all_points)
        
        # Allocate budget among epsilon runs
        budget_per_run = remaining_budget // n_eps
        if budget_per_run < 10:
            # Not enough budget, just return what we have
            return np.array(all_points)
        
        for eps in epsilons:
            if eval_count[0] >= self.budget:
                break
            
            def objective(x):
                # x is clipped by count_problem
                f = count_problem(x)
                return f[0]
            
            def constraint(x):
                f = count_problem(x)
                # Constraint: f1 <= eps, so eps - f1 >= 0
                return eps - f[1]
            
            # We need to be careful: L-BFGS-B with constraints requires the constraint function
            # to return an array of constraint values (>= 0 for inequality).
            # But each call to constraint also counts as an evaluation.
            # This is problematic because L-BFGS-B may call objective and constraint separately.
            
            # Alternative: Use a penalty approach instead of constraints.
            # Minimize f0 + penalty * max(0, f1 - eps)^2
            
            penalty = 100.0
            
            def penalized_objective(x):
                f = count_problem(x)
                return f[0] + penalty * max(0, f[1] - eps) ** 2
            
            # Start from a random point or the best point found so far
            # Let's start from the point with the best f1 among initial points
            idx = np.argmin(F_init[:, 1])
            x0 = X_init[idx].copy()
            
            # Limit the number of function evaluations in L-BFGS-B
            # maxiter is the number of iterations, but each iteration uses multiple function evals
            # Let's set maxiter to something reasonable
            maxiter = max(10, budget_per_run // (2 * dim))
            
            try:
                result = minimize(penalized_objective, x0, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': maxiter, 'maxls': 10})
                if result.success or result.fun < float('inf'):
                    x_best = np.clip(result.x, lo, hi)
                    f_best = count_problem(x_best)
                    all_points.append(x_best)
                    all_objectives.append(f_best)
            except Exception:
                pass
        
        # Return the points (the return value is ignored by the evaluator, but we need to return something)
        if all_points:
            return np.array(all_points)
        else:
            return X_init
# EVOLVE-BLOCK-END
