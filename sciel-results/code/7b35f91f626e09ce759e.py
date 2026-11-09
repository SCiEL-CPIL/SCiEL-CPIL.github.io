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

        eval_count = [0]
        all_points = []
        all_objectives = []

        def count_problem(x):
            eval_count[0] += 1
            x = np.clip(x, lo, hi)
            f = problem(x)
            return f

        def add_point(x, f):
            all_points.append(x.copy())
            all_objectives.append(f.copy())

        # Phase 1: Random initialization with Sobol-like quasi-random points
        n_init = 20
        rng = np.random.default_rng(42)
        
        # Use stratified sampling for better coverage
        n_per_dim = n_init
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            # Stratified: divide [0,1] into n_init equal bins, sample one from each
            bins = np.linspace(0, 1, n_init + 1)
            for i in range(n_init):
                X_init[i, d] = bins[i] + (bins[i+1] - bins[i]) * rng.random()
        X_init = lo + (hi - lo) * X_init
        
        for x in X_init:
            f = count_problem(x)
            add_point(x, f)
        
        F_init = np.array(all_objectives)
        
        # Find extreme points
        f0_min_idx = np.argmin(F_init[:, 0])
        f1_min_idx = np.argmin(F_init[:, 1])
        f0_min = F_init[f0_min_idx, 0]
        f1_min = F_init[f1_min_idx, 1]
        f0_max = F_init[:, 0].max()
        f1_max = F_init[:, 1].max()
        
        # Phase 2: Epsilon-Constraint method
        # We'll use a penalty-based approach with L-BFGS-B
        # Number of epsilon values based on remaining budget
        remaining = self.budget - eval_count[0]
        n_eps = min(6, max(3, remaining // 30))
        
        # Choose epsilon values: focus on the Pareto-relevant range
        # Use log-spaced or linear spaced between f1_min and f1_max
        epsilons = np.linspace(f1_max, f1_min, n_eps)
        
        # For each epsilon, minimize f0 + penalty * max(0, f1 - eps)^2
        # Start from the point that best satisfies the constraint
        for i, eps in enumerate(epsilons):
            if eval_count[0] >= self.budget:
                break
            
            # Find best starting point: minimize f1 among current points, 
            # or use the point closest to satisfying f1 <= eps
            if len(all_objectives) > 0:
                F_curr = np.array(all_objectives)
                # Prefer points where f1 is close to eps from below
                # Use a soft measure: distance to constraint boundary
                dist = np.abs(F_curr[:, 1] - eps)
                # Also prefer lower f0
                combined = dist + 0.1 * F_curr[:, 0]
                x0 = all_points[np.argmin(combined)].copy()
            else:
                x0 = X_init[0].copy()
            
            # Adaptive penalty: start moderate, increase if constraint violated
            penalty = 50.0
            
            def penalized_objective(x):
                f = count_problem(x)
                violation = max(0.0, f[1] - eps)
                return f[0] + penalty * violation ** 2
            
            # Set maxiter based on remaining budget
            budget_left = self.budget - eval_count[0]
            maxiter = max(5, min(50, budget_left // (2 * dim)))
            
            try:
                result = minimize(penalized_objective, x0, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': maxiter, 'maxls': 15, 'ftol': 1e-8})
                
                if result.fun < float('inf') and np.all(np.isfinite(result.x)):
                    x_best = np.clip(result.x, lo, hi)
                    f_best = count_problem(x_best)
                    add_point(x_best, f_best)
                    
                    # If constraint was violated, try again with higher penalty
                    if f_best[1] > eps + 1e-6:
                        penalty *= 5.0
                        budget_left = self.budget - eval_count[0]
                        if budget_left > 10:
                            maxiter2 = max(3, min(30, budget_left // (2 * dim)))
                            try:
                                result2 = minimize(penalized_objective, x_best, method='L-BFGS-B',
                                                   bounds=list(zip(lo, hi)),
                                                   options={'maxiter': maxiter2, 'maxls': 10, 'ftol': 1e-8})
                                if result2.fun < float('inf') and np.all(np.isfinite(result2.x)):
                                    x_best2 = np.clip(result2.x, lo, hi)
                                    f_best2 = count_problem(x_best2)
                                    add_point(x_best2, f_best2)
                            except Exception:
                                pass
            except Exception:
                pass
        
        # Phase 3: Local refinement of best points found
        if eval_count[0] < self.budget:
            F_curr = np.array(all_objectives)
            # Find non-dominated points
            n_pts = len(F_curr)
            nondom_idx = []
            for i in range(n_pts):
                dominated = False
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_curr[j] <= F_curr[i]) and np.any(F_curr[j] < F_curr[i]):
                        dominated = True
                        break
                if not dominated:
                    nondom_idx.append(i)
            
            # Refine up to 3 best non-dominated points
            budget_left = self.budget - eval_count[0]
            n_refine = min(3, len(nondom_idx), budget_left // 20)
            
            for idx in nondom_idx[:n_refine]:
                if eval_count[0] >= self.budget:
                    break
                x0 = all_points[idx].copy()
                
                def refine_objective(x):
                    f = count_problem(x)
                    return f[0] + f[1]  # Sum of objectives for refinement
                
                budget_left = self.budget - eval_count[0]
                maxiter = max(5, min(30, budget_left // (2 * dim)))
                
                try:
                    result = minimize(refine_objective, x0, method='L-BFGS-B',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': maxiter, 'maxls': 10, 'ftol': 1e-9})
                    if result.fun < float('inf') and np.all(np.isfinite(result.x)):
                        x_best = np.clip(result.x, lo, hi)
                        f_best = count_problem(x_best)
                        add_point(x_best, f_best)
                except Exception:
                    pass
        
        if all_points:
            return np.array(all_points)
        else:
            return X_init
# EVOLVE-BLOCK-END
