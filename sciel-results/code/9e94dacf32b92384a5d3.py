import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 0: quick multi-start to seed a rough best (helps AOCC early progress)
        for _ in range(min(10, budget)):
            if not time_ok() or calls >= budget:
                break
            x0 = rng.uniform(-4.5, 4.5, d)
            fx0 = float(func(x0))
            calls += 1
            record(x0, fx0)

        # Reserve budget for final polish
        polish_budget = max(50, int(0.15 * budget))
        cma_limit = budget - polish_budget

        # Phase 1: CMA-ES with restarts
        # CMA-ES adapts the covariance matrix to handle rotated/anisotropic landscapes
        # Population size based on dimension
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        mu = lambda_ // 2
        # Weights: log-weighted
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        
        # Step size parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Restart schedule: each restart gets a portion of the budget
        # We'll do multiple restarts, each with its own CMA-ES instance
        num_restarts = 5
        restart_budget = max(30, (cma_limit - 10) // num_restarts)

        for restart_idx in range(num_restarts):
            if not time_ok() or calls >= cma_limit:
                break

            # Initialize mean and sigma for this restart
            if best_x is None:
                mean = rng.uniform(-4.5, 4.5, d)
            else:
                # Perturb from best with decreasing spread
                spread = 2.0 / (restart_idx + 1)
                mean = np.clip(best_x + spread * rng.standard_normal(d), -5.0, 5.0)
            
            sigma = 1.5 / (restart_idx + 1)
            C = np.eye(d)
            C_sqrt_inv = np.eye(d)
            ps = np.zeros(d)
            pc = np.zeros(d)
            chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))

            # Initial population
            pops = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                x_i = np.clip(mean + sigma * (C_sqrt_inv @ z), -5.0, 5.0)
                pops.append(x_i)

            # Evaluate initial population
            fvals = []
            for x_i in pops:
                if not time_ok() or calls >= cma_limit:
                    break
                fx_i = float(func(x_i))
                calls += 1
                record(x_i, fx_i)
                fvals.append(fx_i)

            if len(fvals) < lambda_:
                continue

            # Sort by fitness
            order = np.argsort(fvals)
            sorted_pops = [pops[i] for i in order]
            sorted_fvals = [fvals[i] for i in order]

            # Track stagnation
            no_improve = 0
            prev_best = best_fx

            while time_ok() and calls < cma_limit:
                # Update mean with weighted recombination
                new_mean = sum(w[i] * sorted_pops[i] for i in range(mu))
                
                # Update covariance matrix
                # Compute step size control
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt_inv.T @ (new_mean - mean))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) / chi_d) < 1.4 + 2 / (d + 1)
                
                # Update evolution path for covariance
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
                
                # Update covariance matrix
                delta_c = 0
                for i in range(mu):
                    y_i = (sorted_pops[i] - mean) / sigma
                    delta_c += w[i] * np.outer(y_i, y_i)
                
                C = (1 + c_1 * (hsig - 1) + c_mu) * C + c_1 * np.outer(pc, pc) - c_1 * C + c_mu * (delta_c - C)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Regularize
                C += 1e-6 * np.eye(d)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1))
                sigma = float(np.clip(sigma, 1e-7, 5.0))
                
                # Update mean
                mean = new_mean
                
                # Check for improvement
                if sorted_fvals[0] < best_fx:
                    best_fx = sorted_fvals[0]
                    best_x = sorted_pops[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Restart if stagnated
                if no_improve > 40:
                    break
                
                # Generate next population
                pops = []
                for i in range(lambda_):
                    z = rng.standard_normal(d)
                    # Use Cholesky decomposition for C_sqrt
                    try:
                        L = np.linalg.cholesky(C)
                        x_i = np.clip(mean + sigma * (L @ z), -5.0, 5.0)
                    except np.linalg.LinAlgError:
                        x_i = np.clip(mean + sigma * z, -5.0, 5.0)
                    pops.append(x_i)
                
                # Evaluate population
                fvals = []
                for x_i in pops:
                    if not time_ok() or calls >= cma_limit:
                        break
                    fx_i = float(func(x_i))
                    calls += 1
                    record(x_i, fx_i)
                    fvals.append(fx_i)
                
                if len(fvals) < lambda_:
                    break
                
                # Sort by fitness
                order = np.argsort(fvals)
                sorted_pops = [pops[i] for i in order]
                sorted_fvals = [fvals[i] for i in order]

        # Phase 2: local polish
        if best_x is not None and time_ok() and calls < budget:
            x = best_x.copy()
            fx = best_fx

            # Coordinate descent with adaptive step
            for _ in range(3):
                if not time_ok() or calls >= budget:
                    break
                for coord in range(d):
                    if not time_ok() or calls >= budget:
                        break
                    for direction in (1, -1):
                        if not time_ok() or calls >= budget:
                            break
                        step = 0.3
                        for _ in range(10):
                            if not time_ok() or calls >= budget:
                                break
                            y = x.copy()
                            y[coord] = np.clip(y[coord] + direction * step, -5.0, 5.0)
                            fy = float(func(y))
                            calls += 1
                            if fy < fx:
                                x = y
                                fx = fy
                                if fx < best_fx:
                                    best_fx = fx
                                    best_x = x.copy()
                                step *= 0.7
                            else:
                                break

            # Adaptive random fine steps with tighter bounds
            sigma = 0.05
            while time_ok() and calls < budget:
                y = np.clip(x + sigma * rng.standard_normal(d), -5.0, 5.0)
                fy = float(func(y))
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.8

        # Safety: ensure a valid result
        if best_x is None:
            x = np.zeros(d)
            best_fx = float(func(x))
            best_x = x.copy()

        return best_x, best_fx
