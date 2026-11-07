"""Combined optimizer: multi-start SLSQP with CMA-ES valley alignment and adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start SLSQP with CMA-ES valley alignment and adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf
        call_count = [0]
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            call_count[0] += 1
            return func(np.asarray(x, dtype=np.float64))

        def time_left():
            return (deadline - time.time()) > 5.0

        def budget_left():
            return (budget - call_count[0]) > 5

        def can_continue():
            return time_left() and budget_left()

        # Phase 1: Initial exploration with structured and random points
        n_initial = min(15, max(8, budget // 40))
        
        structured_points = [np.zeros(dim)]
        for i in range(dim):
            for val in [1.0, 2.0, -1.0, -2.0]:
                pt = np.zeros(dim)
                pt[i] = val
                structured_points.append(pt)
        
        for i in range(n_initial):
            if not can_continue():
                break
            
            if i < len(structured_points):
                x0 = structured_points[i].copy()
            else:
                x0 = rng.uniform(-4.0, 4.0, dim)
            
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # Phase 2: CMA-ES style adaptive search with covariance alignment
        # This phase uses a simplified CMA-ES that adapts to the Rosenbrock valley
        if can_continue():
            # Initialize CMA-ES-like state
            x_mean = best_x.copy()
            sigma = 1.0
            C = np.eye(dim)  # Covariance matrix
            B = np.eye(dim)  # Eigenvectors
            D = np.ones(dim)  # Eigenvalues
            
            # CMA-ES parameters
            pop_size = 2 * (4 + int(np.log(dim)))
            lambda_ = pop_size
            mu = lambda_ // 2  # Number of best solutions for update
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            max_cma_iters = min(20, (budget - call_count[0] - 20) // (lambda_ + dim))
            
            for iter_cma in range(max_cma_iters):
                if not can_continue():
                    break
                
                # Generate population
                ys = []
                xs = []
                fs = []
                
                for i in range(lambda_):
                    if not can_continue():
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x = x_mean + sigma * y
                    x = np.clip(x, -5.0, 5.0)
                    f = wrapped_func(x)
                    ys.append(y)
                    xs.append(x)
                    fs.append(f)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                
                if len(ys) < mu:
                    break
                
                # Sort by fitness
                sorted_idx = np.argsort(fs)
                ys_sorted = [ys[i] for i in sorted_idx[:mu]]
                xs_sorted = [xs[i] for i in sorted_idx[:mu]]
                
                # Update mean
                y_w = sum(weights[i] * ys_sorted[i] for i in range(mu))
                x_w = sum(weights[i] * xs_sorted[i] for i in range(mu))
                
                # Update step size control
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * (B @ (D * y_w))
                
                # Update evolution path
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * y_w
                
                # Update covariance matrix
                delta = 0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iter_cma + 1) / dim)) < (1 + 2 / (dim + 1)) * chi_n else 1
                C = (1 - c_1 - c_mu + delta * c_1 * c_sigma * (2 - c_sigma)) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                C += c_mu * sum(weights[i] ** 2 * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu))
                
                # Ensure positive semi-definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                
                # Update mean
                x_mean = x_w
                x_mean = np.clip(x_mean, -5.0, 5.0)
                
                # Adapt sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 0.01, 10.0)

        # Phase 3: Primary SLSQP from the best point
        remaining = budget - call_count[0] - 15
        if remaining > 10 and can_continue():
            max_iter = max(15, min(60, remaining // (dim + 1)))
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: Multi-start local refinement with SLSQP
        n_local_starts = min(4, max(2, (budget - call_count[0] - 20) // 50))
        
        for j in range(n_local_starts):
            if not can_continue():
                break
            
            # Create perturbations around the best point with decreasing scale
            scale = 1.0 / (j + 1)
            x0 = best_x + rng.normal(0, scale, dim)
            x0 = np.clip(x0, -5.0, 5.0)
            
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            remaining = budget - call_count[0] - 15
            if remaining > 5 and can_continue():
                max_iter = max(10, min(40, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    # Fallback to Nelder-Mead if SLSQP fails
                    remaining = budget - call_count[0] - 10
                    if remaining > 5 and can_continue():
                        try:
                            res = minimize(
                                wrapped_func,
                                x0,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': min(60, remaining // 3),
                                    'xatol': 1e-10,
                                    'fatol': 1e-10,
                                    'adaptive': True,
                                    'disp': False
                                }
                            )
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = res.x.copy()
                        except Exception:
                            pass

        # Phase 5: Multi-scale refinement around the best point
        if best_x is not None and can_continue():
            scales = [0.5, 0.25, 0.1, 0.05]
            for scale in scales:
                if not can_continue():
                    break
                
                # Generate perturbation around best point
                perturbation = rng.normal(0, scale, dim)
                start_point = np.clip(best_x + perturbation, -5.0, 5.0)
                
                remaining = budget - call_count[0] - 15
                if remaining > 5 and can_continue():
                    max_iter = max(10, min(30, remaining // (dim + 1)))
                    try:
                        res = minimize(
                            wrapped_func,
                            start_point,
                            method='SLSQP',
                            bounds=bounds,
                            options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # Phase 6: Final high-precision polish
        if best_x is not None and can_continue():
            remaining = budget - call_count[0] - 5
            if remaining > 5:
                max_iter_final = max(20, min(100, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter_final, 'ftol': 1e-15, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    # Fallback to tight Nelder-Mead
                    remaining = budget - call_count[0] - 3
                    if remaining > 3 and can_continue():
                        try:
                            res = minimize(
                                wrapped_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': min(80, remaining // 2),
                                    'xatol': 1e-12,
                                    'fatol': 1e-12,
                                    'adaptive': True,
                                    'disp': False
                                }
                            )
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = res.x.copy()
                        except Exception:
                            pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
