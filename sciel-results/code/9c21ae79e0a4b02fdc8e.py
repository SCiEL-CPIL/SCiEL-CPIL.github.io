import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and L-BFGS-B polish for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Safety margin: 330s limit, stop at 325s
        deadline = time.time() + 325

        best_x = None
        best_f = np.inf
        calls = 0

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if x.shape != (dim,):
                x = x.reshape(-1)
            if len(x) != dim:
                x = x[:dim]
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def polish(x_start, maxfev=20):
            """Run L-BFGS-B polish from a starting point."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            try:
                res = minimize(
                    wrapped_func,
                    x_start.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfev': maxfev,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 1: Initial sampling with structured points
        n_initial = min(30, budget // 10)
        
        # Structured initial points - more diverse
        init_points = [
            np.zeros(dim),
            np.full(dim, -5.0),
            np.full(dim, 5.0),
            np.full(dim, -3.0),
            np.full(dim, 3.0),
            np.full(dim, -1.0),
            np.full(dim, 1.0),
            np.full(dim, -2.0),
            np.full(dim, 2.0),
            np.full(dim, -4.0),
            np.full(dim, 4.0),
        ]
        # Add random points
        for i in range(max(0, n_initial - len(init_points))):
            init_points.append(rng.uniform(-5, 5, dim))

        for x in init_points:
            if calls >= budget or time.time() > deadline:
                break
            wrapped_func(x)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Quick polish from initial best
        polish(best_x, maxfev=max(15, (budget - calls) // 8))

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-start CMA-ES with adaptive restarts
        def run_cmaes(x0, maxevals, sigma0=None):
            """Run a simple CMA-ES implementation."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            
            n = dim
            if sigma0 is None:
                sigma0 = 1.0
            
            # Population size - slightly larger for better exploration
            lambda_pop = max(4 + int(3 * np.log(n)), 16)
            
            # CMA-ES parameters
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            
            # State variables
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Track best
            local_best_x = x0.copy()
            local_best_f = wrapped_func(x0)
            
            eval_count = 0
            stagnation = 0
            last_improvement = 0
            best_at_start = local_best_f
            
            while eval_count < maxevals and calls < budget and time.time() < deadline:
                # Generate population
                pop = []
                for i in range(lambda_pop):
                    # Sample from N(x_mean, sigma^2 * C)
                    try:
                        B = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    except np.linalg.LinAlgError:
                        B = np.eye(n)
                    z = rng.standard_normal(n)
                    x_new = x_mean + sigma * (B @ z)
                    x_new = np.clip(x_new, -5, 5)
                    pop.append(x_new)
                
                # Evaluate population
                pop_f = []
                for x in pop:
                    if calls >= budget or time.time() > deadline:
                        break
                    f = wrapped_func(x)
                    pop_f.append(f)
                    eval_count += 1
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x.copy()
                        last_improvement = eval_count
                
                if len(pop_f) < lambda_pop:
                    break
                
                # Sort by fitness
                indices = np.argsort(pop_f)
                pop = [pop[i] for i in indices]
                pop_f = [pop_f[i] for i in indices]
                
                # Update mean
                x_mean_new = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
                
                # Update step size control
                try:
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                    D = np.diag(B)
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / D @ (B.T @ (x_mean_new - x_mean) / sigma)
                    
                    if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (eval_count + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    
                    # Update covariance
                    delta_sig = (1 - c_sigma) ** 2 * (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1)
                    rank_mu = np.outer(x_mean_new - x_mean, x_mean_new - x_mean)
                    rank_one = np.outer(x_mean - x_mean_new, x_mean - x_mean_new)
                    
                    C = (1 + c_1 + c_mu - c_c) * C + c_1 * rank_one + c_mu * rank_mu
                    C = (C + C.T) / 2  # Symmetrize
                except Exception:
                    sigma *= 0.9  # Reduce step size if covariance is degenerate
                
                x_mean = x_mean_new
                x_mean = np.clip(x_mean, -5, 5)
                
                # Check for stagnation
                if local_best_f == best_f or (local_best_f - best_f) < 1e-10:
                    stagnation += 1
                else:
                    stagnation = 0
                
                if stagnation > 35 or (eval_count - last_improvement > 25 and eval_count > 40):
                    break
            
            # Update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            return local_best_x, local_best_f

        # Run multiple CMA-ES instances from different starting points
        n_restarts = min(8, (budget - calls) // 35)
        
        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Starting point: mix of best found and random
            if r == 0:
                x0 = best_x.copy()
                sigma0 = 0.4
            elif r == 1:
                x0 = best_x + rng.normal(0, 0.6, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.8
            elif r == 2:
                x0 = best_x + rng.normal(0, 1.0, dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.2
            elif r == 3:
                x0 = rng.uniform(-2, 2, dim)
                sigma0 = 1.5
            elif r == 4:
                x0 = rng.uniform(-3, 3, dim)
                sigma0 = 1.8
            elif r == 5:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 2.2
            elif r == 6:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.5
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 3.0
            
            # Allocate remaining budget
            remaining = budget - calls
            maxevals = max(20, remaining // (n_restarts - r))
            
            run_cmaes(x0, maxevals, sigma0)
            
            # Polish after each restart
            if calls < budget - 15 and time.time() < deadline:
                polish(best_x, maxfev=max(8, (budget - calls) // 8))

        # Phase 3: Final high-precision polish
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                # Multiple polish passes with decreasing tolerance
                polish_configs = [
                    (100, remaining, 1e-15, 1e-12),
                    (60, max(15, remaining // 3), 1e-16, 1e-13),
                    (40, max(10, remaining // 5), 1e-17, 1e-14),
                ]
                for maxiter, maxfev, ftol, gtol in polish_configs:
                    if calls >= budget or time.time() > deadline:
                        break
                    try:
                        minimize(
                            wrapped_func,
                            best_x.copy(),
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': maxiter,
                                'maxfev': maxfev,
                                'ftol': ftol,
                                'gtol': gtol,
                                'disp': False
                            }
                        )
                    except Exception:
                        pass
        
        # Final check
        if best_x is not None:
            # One last evaluation to ensure we have the best
            if calls < budget and time.time() < deadline:
                val = wrapped_func(best_x)
        
        return best_x, best_f
