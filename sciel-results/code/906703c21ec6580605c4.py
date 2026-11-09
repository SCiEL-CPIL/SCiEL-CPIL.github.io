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

        def bounded_minimize(obj_func, x0, bounds_list, max_iter=30, gtol=1e-8):
            """Robust bounded minimization with fallback."""
            try:
                res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds_list,
                               options={'maxiter': max_iter, 'gtol': gtol})
                return res.x
            except Exception:
                pass
            try:
                res = minimize(obj_func, x0, method='Nelder-Mead',
                               options={'maxiter': max_iter * 3, 'xatol': 1e-8, 'fatol': 1e-10})
                # Project back to bounds
                x_res = np.clip(res.x, lo, hi)
                return x_res
            except Exception:
                return x0

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
            def obj1(x):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                return f[0]

            x0_1 = np.random.uniform(lo, hi)
            try:
                res1_x = bounded_minimize(obj1, x0_1, bounds, max_iter=25, gtol=1e-7)
                if self.eval_count < self.budget:
                    f1 = evaluate(res1_x)
                    add_to_archive(res1_x, f1)
            except Exception:
                pass

        if self.eval_count < self.budget - 15:
            def obj2(x):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                return f[1]

            x0_2 = np.random.uniform(lo, hi)
            try:
                res2_x = bounded_minimize(obj2, x0_2, bounds, max_iter=25, gtol=1e-7)
                if self.eval_count < self.budget:
                    f2 = evaluate(res2_x)
                    add_to_archive(res2_x, f2)
            except Exception:
                pass

        # Get current bounds of f1 and f2 from archive
        if len(self.archive) == 0:
            return self.archive

        f_vals = np.array([f for _, f in self.archive])
        f1_min, f1_max = f_vals[:, 0].min(), f_vals[:, 0].max()
        f2_min, f2_max = f_vals[:, 1].min(), f_vals[:, 1].max()
        f1_range = f1_max - f1_min + 1e-10
        f2_range = f2_max - f2_min + 1e-10

        # Phase 2: Epsilon-Constraint Method with smooth surrogate
        remaining = self.budget - self.eval_count
        n_steps = max(10, min(35, remaining // 5))

        # Create a grid of epsilon values
        epsilons = np.linspace(f2_min, f2_max, n_steps)

        for eps in epsilons:
            if self.eval_count >= self.budget - 5:
                break
            if time.time() > deadline:
                break

            # Scale-aware penalty weight
            mu = f1_range / (f2_range + 1e-10) * 5.0
            k = 8.0

            def penalized_obj(x, eps=eps, mu=mu, k=k):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                z = f[1] - eps
                # Smooth penalty: mu * softplus(z) / k, approximates max(0,z)
                penalty = mu * np.log1p(np.exp(np.clip(k * z, -50, 50))) / k
                return f[0] + penalty

            # Multi-start: try from archive point closest to eps AND a random point
            best_x = None
            best_f = None

            candidates = []
            if len(self.archive) > 0:
                f2_vals = np.array([f[1] for _, f in self.archive])
                idx = np.argmin(np.abs(f2_vals - eps))
                candidates.append(self.archive[idx][0].copy())
            candidates.append(np.random.uniform(lo, hi))

            for x0 in candidates:
                if self.eval_count >= self.budget - 3:
                    break
                if time.time() > deadline:
                    break
                try:
                    res_x = bounded_minimize(penalized_obj, x0, bounds, max_iter=20, gtol=1e-7)
                    if self.eval_count < self.budget:
                        f_res = evaluate(res_x)
                        if best_f is None or f_res[0] < best_f[0]:
                            best_x = res_x
                            best_f = f_res
                except Exception:
                    pass

            if best_x is not None and best_f is not None:
                # Only add if it's reasonably close to satisfying the constraint
                if best_f[1] <= eps + 0.15 * f2_range:
                    add_to_archive(best_x, best_f)

        # Phase 3: Final high-precision polish pass
        # Refine each archive point with adaptive weights based on position on the front
        if self.eval_count < self.budget - 5 and time.time() < deadline:
            # Sort archive by f1
            sorted_archive = sorted(self.archive, key=lambda xf: xf[1][0])
            n_arch = len(sorted_archive)

            for i, (x_arch, f_arch) in enumerate(sorted_archive):
                if self.eval_count >= self.budget - 3:
                    break
                if time.time() > deadline:
                    break

                # Adaptive weight: points with low f1 get high weight on f1,
                # points with low f2 get high weight on f2
                # Use normalized position
                if n_arch > 1:
                    t = i / (n_arch - 1)  # 0 = best f1, 1 = worst f1
                else:
                    t = 0.5

                w1 = 1.0 - t  # weight on f1
                w2 = t        # weight on f2

                def local_obj(x, w1=w1, w2=w2):
                    if self.eval_count >= self.budget:
                        return 0.0
                    f = problem(x)
                    self.eval_count += 1
                    return w1 * f[0] / f1_range + w2 * f[1] / f2_range

                try:
                    res_x = bounded_minimize(local_obj, x_arch, bounds, max_iter=25, gtol=1e-9)
                    if self.eval_count < self.budget:
                        f_local = evaluate(res_x)
                        add_to_archive(res_x, f_local)
                except Exception:
                    pass

        return self.archive
# EVOLVE-BLOCK-END
