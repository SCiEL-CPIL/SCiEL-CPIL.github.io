"""Surrogate-assisted CMA-ES for BBOB-style Schwefel f20 on [-5,5]^5."""
import numpy as np
import time
from scipy.interpolate import Rbf

class Optimizer:
    """Surrogate-assisted CMA-ES for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        def time_exceeded():
            return time.time() > deadline

        best_f = np.inf
        best_x = np.zeros(dim)
        evals = 0

        # Wrap func to track evaluations and handle clipping
        def safe_func(x):
            nonlocal evals
            if evals >= budget:
                return np.inf
            x_clip = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x_clip))
                evals += 1
                if evals >= budget:
                    return val
                return val
            except Exception:
                evals += 1
                return np.inf

        # Phase 1: CMA-ES with surrogate guidance
        # CMA-ES parameters
        lambda_pop = 12  # population size
        c_sigma = 0.5
        c_sigma_min = 1e-4
        sigma_min = 1e-6
        sigma_max = 5.0

        # CMA-ES state
        m = np.zeros(dim)  # mean
        sigma = 1.5
        C = np.eye(dim)  # covariance
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Pre-compute
        def mat_update():
            nonlocal C_inv, C_sqrt
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                # Fallback
                C += 1e-8 * np.eye(dim)
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_inv = np.linalg.inv(C)

        mat_update()

        # Surrogate
        X_samples = []
        Y_samples = []

        def update_surrogate():
            if len(X_samples) < dim + 1:
                return None
            try:
                rbf = Rbf(*[X_samples.T[i] for i in range(dim)],
                          Y_samples, function='multiquadric')
                return rbf
            except Exception:
                return None

        # Phase 1: CMA-ES with periodic surrogate-guided candidates
        cma_evals = 0
        max_cma_evals = min(400, budget - 20)
        surrogate_interval = 30

        for gen in range(max_cma_evals // lambda_pop + 10):
            if time_exceeded() or evals >= budget:
                break

            # Generate population
            pops = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                # Use C_sqrt to transform
                xi = m + sigma * (C_sqrt @ z)
                xi = np.clip(xi, -5.0, 5.0)
                pops.append((xi, safe_func(xi)))

            # Sort by fitness
            pops.sort(key=lambda t: t[1])

            # Update best
            if pops[0][1] < best_f:
                best_f = pops[0][1]
                best_x = pops[0][0].copy()

            # Add to surrogate data
            for xi, fi in pops:
                X_samples.append(xi)
                Y_samples.append(fi)
                cma_evals += 1

            # CMA-ES update
            # Weighted mean
            mu = lambda_pop // 2
            weights = np.zeros(lambda_pop)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()

            # Update mean
            new_m = np.zeros(dim)
            for i in range(lambda_pop):
                new_m += weights[i] * pops[i][0]
            m = new_m

            # Covariance update
            diff = np.zeros((dim, mu))
            for i in range(mu):
                diff[:, i] = (pops[i][0] - m) / sigma
            H = diff @ np.diag(weights) @ diff.T
            c1 = (2 / ((dim + 1) ** 2 + 2))
            cmu = 2 * (weights.sum() ** 2 - 1 / weights.sum()) / ((dim + 2) ** 2 - 4)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(m - m, m - m) + np.eye(dim)) + cmu * H
            C = (C + C.T) / 2

            # Sigma update (simplified)
            sigma *= np.exp(c_sigma * (np.sqrt(sum(weights[:mu] ** 2)) *
                                        np.linalg.norm(np.array([pops[i][0] for i in range(mu)]) @ np.diag(weights) @ C_sqrt.T) / sigma - 1))
            sigma = np.clip(sigma, sigma_min, sigma_max)

            mat_update()

            # Periodically try surrogate-guided candidate
            if cma_evals % surrogate_interval == 0 and len(X_samples) > 10:
                rbf = update_surrogate()
                if rbf is not None and evals < budget - 5:
                    # Try a few surrogate-guided points
                    for trial in range(3):
                        # Random point in the box
                        x_try = rng.uniform(-5, 5, size=dim)
                        # Evaluate surrogate
                        try:
                            f_surr = float(rbf(*x_try))
                        except Exception:
                            continue
                        # If surrogate predicts a good value, evaluate
                        if f_surr < best_f * 0.9 + 1:
                            f_real = safe_func(x_try)
                            if f_real < best_f:
                                best_f = f_real
                                best_x = x_try.copy()
                            X_samples.append(x_try)
                            Y_samples.append(f_real)
                            if time_exceeded() or evals >= budget:
                                break

        # Phase 2: Local refinement with Nelder-Mead from best point
        if evals < budget and not time_exceeded():
            from scipy.optimize import minimize
            remaining = budget - evals
            if remaining > 20:
                try:
                    res = minimize(safe_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': min(remaining, 50),
                                            'xatol': 1e-6, 'fatol': 1e-6})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
