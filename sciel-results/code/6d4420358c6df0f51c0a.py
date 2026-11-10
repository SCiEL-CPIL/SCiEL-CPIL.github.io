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
            return np.array(f, dtype=float)

        def add_point(x, f):
            all_points.append(np.array(x, dtype=float).copy())
            all_objectives.append(np.array(f, dtype=float).copy())

        # Phase 1: Latin Hypercube Sampling for diverse initialization
        # Use a larger initial sample to better estimate the Pareto front range
        n_init = min(30, self.budget // 10)
        rng = np.random.default_rng(42)
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X_init[:, d] = (perm + rng.random(n_init)) / n_init
        X_init = lo + (hi - lo) * X_init

        for x in X_init:
            f = count_problem(x)
            add_point(x, f)

        F_init = np.array(all_objectives)

        # Identify nadir and ideal points for scaling/epsilon selection
        f1_min = F_init[:, 1].min()
        f1_max = F_init[:, 1].max()
        f0_min = F_init[:, 0].min()
        f0_max = F_init[:, 0].max()

        # Phase 2: Augmented Epsilon-Constraint Method
        # We minimize f0 subject to f1 <= eps.
        # We use an augmented penalty method: min f0 + mu * max(0, f1 - eps)
        
        remaining = self.budget - eval_count[0]
        # Allocate budget: ~60% for main sweep, ~40% for refinement
        # Number of epsilon levels depends on budget
        n_eps = min(20, max(10, remaining // 15))
        
        # Epsilon values: linearly spaced from f1_max to f1_min
        # Ensure we cover the full range
        epsilons = np.linspace(f1_max, f1_min, n_eps)
        
        # Sort epsilons from high to low (relaxed to tight) to help warm-starting
        # Actually, it's often better to go from tight to loose or vice versa?
        # Let's stick to linear spacing. We will select starting points intelligently.

        for i, eps in enumerate(epsilons):
            if eval_count[0] >= self.budget:
                break

            # Choose starting point
            # Heuristic: Find existing points with f1 closest to eps.
            # If any have f1 <= eps, prefer those.
            if len(all_objectives) > 0:
                F_curr = np.array(all_objectives)
                # Calculate distance in objective space, weighting f1 difference heavily
                # We want f1 close to eps and f0 low.
                # Simple metric: |f1 - eps| + 0.1 * f0
                dists = np.abs(F_curr[:, 1] - eps)
                # Prefer feasible points
                feasible_mask = F_curr[:, 1] <= eps + 1e-6
                if np.any(feasible_mask):
                    # Among feasible, pick lowest f0
                    feasible_indices = np.where(feasible_mask)[0]
                    x0_idx = feasible_indices[np.argmin(F_curr[feasible_indices, 0])]
                else:
                    # Pick closest to eps
                    x0_idx = np.argmin(dists)
                x0 = all_points[x0_idx].copy()
            else:
                x0 = X_init[0].copy()

            # Define augmented objective
            mu = 10.0
            
            def augmented_objective(x):
                f = count_problem(x)
                penalty = mu * max(0.0, f[1] - eps)
                return f[0] + penalty

            # Optimization settings
            budget_left = self.budget - eval_count[0]
            # Limit iterations to avoid wasting budget
            maxiter = max(10, min(50, budget_left // 5))
            
            try:
                result = minimize(augmented_objective, x0, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': maxiter, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-10})

                if result.fun < float('inf') and np.all(np.isfinite(result.x)):
                    x_best = np.clip(result.x, lo, hi)
                    f_best = count_problem(x_best)
                    add_point(x_best, f_best)
                    
                    # If significantly infeasible, try a second round with higher penalty
                    if f_best[1] > eps + 1e-3 and eval_count[0] < self.budget:
                        mu = 100.0
                        def augmented_objective2(x):
                            f = count_problem(x)
                            penalty = mu * max(0.0, f[1] - eps)
                            return f[0] + penalty
                            
                        budget_left = self.budget - eval_count[0]
                        maxiter2 = max(5, min(30, budget_left // 5))
                        try:
                            result2 = minimize(augmented_objective2, x_best, method='L-BFGS-B',
                                               bounds=list(zip(lo, hi)),
                                               options={'maxiter': maxiter2, 'maxls': 10, 'ftol': 1e-12, 'gtol': 1e-10})
                            if result2.fun < float('inf') and np.all(np.isfinite(result2.x)):
                                x_best2 = np.clip(result2.x, lo, hi)
                                f_best2 = count_problem(x_best2)
                                add_point(x_best2, f_best2)
                        except Exception:
                            pass
            except Exception:
                pass

        # Phase 3: Local Refinement of Non-Dominated Points
        # Identify non-dominated points
        if len(all_objectives) > 0:
            F_curr = np.array(all_objectives)
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
            
            # Refine up to 10 best non-dominated points using weighted sum minimization
            budget_left = self.budget - eval_count[0]
            if budget_left > 0:
                # Select a subset of non-dominated points to refine
                # Prioritize points that are "spread out"
                n_refine = min(10, len(nondom_idx), budget_left // 10)
                
                # Random weights for weighted sum to explore different parts of the front
                rng_refine = np.random.default_rng(123)
                
                for k in range(n_refine):
                    if eval_count[0] >= self.budget:
                        break
                    
                    # Pick a non-dominated point
                    idx = nondom_idx[k % len(nondom_idx)]
                    x0 = all_points[idx].copy()
                    
                    # Use a random weight for the second objective to steer the search
                    w = rng_refine.uniform(0.1, 1.0)
                    
                    def refine_objective(x):
                        f = count_problem(x)
                        return f[0] + w * f[1]
                    
                    budget_left = self.budget - eval_count[0]
                    maxiter = max(5, min(30, budget_left // 5))
                    
                    try:
                        result = minimize(refine_objective, x0, method='L-BFGS-B',
                                           bounds=list(zip(lo, hi)),
                                           options={'maxiter': maxiter, 'maxls': 10, 'ftol': 1e-12, 'gtol': 1e-10})
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
