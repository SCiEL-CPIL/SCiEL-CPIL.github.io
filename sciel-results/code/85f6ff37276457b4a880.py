"""Multi-objective optimizer for RE21 using Epsilon-Constraint with smooth surrogate and local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []  # List of (x, f) tuples
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 280.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        def add_to_archive(x, f):
            """Add point to archive, removing dominated points."""
            new_archive = []
            for x_old, f_old in self.archive:
                # If f_old is dominated by f, discard f_old
                if np.all(f <= f_old) and np.any(f < f_old):
                    continue
                new_archive.append((x_old, f_old))
            new_archive.append((x, f))
            self.archive = new_archive

        def evaluate(x):
            """Evaluate problem at x, tracking budget."""
            f = problem(x)
            self.eval_count += 1
            return f

        # Phase 1: Initialization - sample points to understand the landscape
        n_init = min(25, self.budget // 4)
        X_init = np.random.random((n_init, self.dim))
        X_init = lo + span * X_init

        for x in X_init:
            if self.eval_count >= self.budget - 10:
                break
            f = evaluate(x)
            add_to_archive(x, f)

        # Find approximate optima for each objective individually
        if self.eval_count < self.budget - 15:
            # Minimize f1
            def obj1(x):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                return f[0]

            x0_1 = np.random.uniform(lo, hi)
            try:
                res1 = minimize(obj1, x0_1, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 20, 'maxfun': 20, 'gtol': 1e-6})
                if self.eval_count < self.budget:
                    f1 = evaluate(res1.x)
                    add_to_archive(res1.x, f1)
            except Exception:
                pass

        if self.eval_count < self.budget - 15:
            # Minimize f2
            def obj2(x):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                return f[1]

            x0_2 = np.random.uniform(lo, hi)
            try:
                res2 = minimize(obj2, x0_2, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 20, 'maxfun': 20, 'gtol': 1e-6})
                if self.eval_count < self.budget:
                    f2 = evaluate(res2.x)
                    add_to_archive(res2.x, f2)
            except Exception:
                pass

        # Get current bounds of f1 and f2 from archive
        if len(self.archive) == 0:
            return self.archive

        f_vals = np.array([f for _, f in self.archive])
        f1_min, f1_max = f_vals[:, 0].min(), f_vals[:, 0].max()
        f2_min, f2_max = f_vals[:, 1].min(), f_vals[:, 1].max()

        # Phase 2: Epsilon-Constraint Method with smooth surrogate
        # We sweep epsilon for the second objective and minimize the first.
        remaining = self.budget - self.eval_count
        n_steps = max(8, min(30, remaining // 6))  # Allow ~6 evals per step

        # Create a grid of epsilon values, focusing on the Pareto-relevant range
        epsilons = np.linspace(f2_min, f2_max, n_steps)

        for eps in epsilons:
            if self.eval_count >= self.budget - 5:
                break
            if time.time() > deadline:
                break

            # Penalized objective: f1 + mu * max(0, f2 - eps)^2
            # Use smooth approximation: mu * (f2 - eps)^2 / (1 + exp(-k*(f2-eps)))
            # Actually, let's use a simple quadratic penalty for smoothness
            mu = 10.0  # Penalty weight
            k = 10.0   # Sharpness of soft constraint

            def penalized_obj(x):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                # Smooth penalty: use softplus-like function
                # penalty = mu * softplus(f2 - eps) where softplus(z) = log(1+exp(z))
                # This is smooth and approximates max(0, z) for large k
                z = f[1] - eps
                penalty = mu * np.log1p(np.exp(np.clip(z, -50, 50)))
                return f[0] + penalty

            # Initial guess: from archive or random
            # Try to start near points that have f2 close to eps
            x0 = np.random.uniform(lo, hi)
            if len(self.archive) > 0:
                # Find archive point with f2 closest to eps
                f2_vals = np.array([f[1] for _, f in self.archive])
                idx = np.argmin(np.abs(f2_vals - eps))
                x0 = self.archive[idx][0].copy()

            try:
                res = minimize(
                    penalized_obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'maxfun': 30, 'gtol': 1e-6}
                )
                if self.eval_count < self.budget:
                    f_res = evaluate(res.x)
                    # Only add if it's reasonably close to satisfying the constraint
                    if f_res[1] <= eps + 0.1 * (f2_max - f2_min + 1e-10):
                        add_to_archive(res.x, f_res)
            except Exception:
                pass

        # Phase 3: Local refinement of archive points
        # For each archive point, do a local search to improve
        if self.eval_count < self.budget - 10 and time.time() < deadline:
            archive_copy = list(self.archive)
            for x_arch, f_arch in archive_copy:
                if self.eval_count >= self.budget - 5:
                    break
                if time.time() > deadline:
                    break

                # Local refinement using L-BFGS-B on a weighted sum
                # Weight based on position in the front
                def local_obj(x):
                    if self.eval_count >= self.budget:
                        return 0.0
                    f = problem(x)
                    self.eval_count += 1
                    # Weighted sum with weights based on archive position
                    return 0.5 * f[0] + 0.5 * f[1]

                try:
                    res_local = minimize(
                        local_obj,
                        x_arch,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 15, 'maxfun': 15, 'gtol': 1e-7}
                    )
                    if self.eval_count < self.budget:
                        f_local = evaluate(res_local.x)
                        add_to_archive(res_local.x, f_local)
                except Exception:
                    pass

        return self.archive
# EVOLVE-BLOCK-END
