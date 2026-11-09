"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start with covariance-matrix adaptation and coordinate descent."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
                calls_used += 1
                return val
            except Exception:
                calls_used += 1
                return np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline

        def coordinate_descent(x, fx, max_iters=3):
            """Simple coordinate descent with adaptive step size."""
            nonlocal calls_used
            x = x.copy()
            fx = fx
            step = 2.0
            for _ in range(max_iters):
                if calls_used >= budget or not time_ok():
                    break
                improved = False
                for d in range(dim):
                    if calls_used >= budget or not time_ok():
                        break
                    best_step = 0
                    best_fx = fx
                    for sign in [1, -1]:
                        y = x.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy < best_fx:
                            best_fx = fy
                            best_step = sign * step
                    if best_step != 0:
                        x[d] += best_step
                        fx = best_fx
                        improved = True
                        update_best(x, fx)
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break
            return x, fx

        def cma_es(x, fx, max_iters=100):
            """Simplified CMA-ES with population size 4."""
            nonlocal calls_used
            x = x.copy()
            fx = fx
            sigma = 1.0
            C = np.eye(dim)  # Covariance matrix
            C_inv = np.eye(dim)
            lam = 4  # Population size
            
            for it in range(max_iters):
                if calls_used >= budget or not time_ok():
                    break
                
                # Generate offspring
                offspring = []
                offspring_f = []
                for j in range(lam):
                    if calls_used >= budget or not time_ok():
                        break
                    z = rng.standard_normal(dim)
                    # Transform using C
                    y = x + sigma * np.linalg.cholesky(C).T @ z
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy < np.inf:
                        offspring.append(y)
                        offspring_f.append(fy)
                
                if not offspring:
                    break
                
                # Sort by fitness
                order = np.argsort(offspring_f)
                mu = lam // 2
                selected = [offspring[i] for i in order[:mu]]
                selected_f = [offspring_f[i] for i in order[:mu]]
                
                # Update mean
                new_x = np.mean(selected, axis=0)
                new_fx = min(selected_f)
                
                # Update sigma
                sigma *= 1.05 if new_fx < fx else 0.95
                sigma = max(0.001, min(sigma, 5.0))
                
                # Update covariance (simplified)
                if new_fx < fx:
                    diff = (new_x - x) / sigma
                    C += 0.01 * np.outer(diff, diff)
                    C = (C + C.T) / 2  # Symmetrize
                    # Renormalize
                    C = C / np.trace(C) * dim
                    C_inv = np.linalg.inv(C)
                
                x, fx = new_x, new_fx
                update_best(x, fx)
                
                if sigma < 0.005:
                    break
            
            return x, fx

        # Phase 1: Coarse random sampling
        n_coarse = min(20, budget // 15)
        coarse_points = []
        coarse_vals = []
        for _ in range(n_coarse):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx < np.inf:
                coarse_points.append(x)
                coarse_vals.append(fx)

        if not coarse_points:
            x0 = np.zeros(dim)
            fx0 = safe_call(x0)
            if fx0 < np.inf:
                best_x = x0
                best_f = fx0
            else:
                best_x = np.zeros(dim)
                best_f = 0.0
            return best_x, best_f

        order = np.argsort(coarse_vals)
        k = min(4, len(coarse_points))
        top_indices = order[:k]

        best_x = coarse_points[top_indices[0]].copy()
        best_f = coarse_vals[top_indices[0]]

        # Phase 2: Refinement from top starts
        for i, start_idx in enumerate(top_indices):
            if calls_used >= budget or not time_ok():
                break

            x = coarse_points[start_idx].copy()
            fx = coarse_vals[start_idx]

            # Coordinate descent first for quick progress
            x, fx = coordinate_descent(x, fx, max_iters=2)

            # Then CMA-ES for fine-tuning
            remaining = budget - calls_used
            if remaining > 20:
                max_cma = min(80, remaining // 4)
                x, fx = cma_es(x, fx, max_iters=max_cma)

        return best_x, best_f
# EVOLVE-BLOCK-END
