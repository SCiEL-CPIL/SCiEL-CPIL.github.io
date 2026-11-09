import numpy as np
import time


class Optimizer:
    """Two-stage optimizer: LHS bracketing then CMA-ES for f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def cma_es(x0, f0, sigma0, max_iter=200):
            nonlocal calls_made, best_x, best_f
            n = dim
            pop_size = int(4 + 3 * np.log(n))
            pop_size = max(pop_size, n + 1)
            
            # CMA-ES parameters
            alpha_mu = 0.5
            mu = pop_size // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            nu = 1 / (weights[0] ** 2 - 1)
            
            # Step size control
            c_sigma = (mu_eff + 2) / ((n + 1) ** 2 + mu_eff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = (2 - mu_eff / (n + 1)) / ((n + 2) ** 2 + mu_eff)
            c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((n + 2) ** 2 + mu_eff))
            
            mu_eff = 1 / sum(weights ** 2)
            
            # Initialize
            mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Initial population
            population = np.zeros((pop_size, n))
            fvals = np.zeros(pop_size)
            
            for i in range(pop_size):
                if not time_ok():
                    break
                z = np.random.randn(n)
                # Approximate transformation using current C
                # For simplicity, use diagonal approximation initially
                y = B @ (D * z)
                x = mean + sigma * y
                fvals[i] = safe_call(x)
                population[i] = x
            
            for it in range(max_iter):
                if not time_ok():
                    break
                
                # Sort by fitness
                order = np.argsort(fvals)
                population = population[order]
                fvals = fvals[order]
                
                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = population[0].copy()
                
                # Check convergence
                if np.std(fvals[:mu]) < 1e-10 * (1 + abs(fvals[0])):
                    break
                
                # Compute weighted mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (population[i] - mean) / sigma
                
                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_w
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update mean
                mean += c_c * (y_w / c_sigma - p_sigma)
                
                # Update p_c
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                
                # Update C
                delta_hsig = (1 + (1 - c_c) * np.linalg.norm(p_c) ** 2 / (n * (1 - (1 - c_c) ** (it + 1)))) * (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1))
                C = (1 + c_1 + c_mu - c_1 / n) * C
                for i in range(mu):
                    y_i = (population[i] - mean) / sigma
                    C -= c_1 * np.outer(y_i, y_i)
                    C += c_mu * weights[i] * np.outer(y_i, y_i)
                
                # Ensure C is positive definite (simple fix)
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update B and D
                eigvals2, eigvecs2 = np.linalg.eigh(C)
                eigvals2 = np.clip(eigvals2, 1e-12, None)
                B = eigvecs2
                D = np.sqrt(eigvals2)
                
                # Generate new population
                for i in range(pop_size):
                    if not time_ok():
                        break
                    z = np.random.randn(n)
                    y = B @ (D * z)
                    x = mean + sigma * y
                    fvals[i] = safe_call(x)
                    population[i] = x
            
            return best_x, best_f

        # Phase 1: Latin Hypercube Sampling (30 points)
        n_samples = 30
        rng = np.random.RandomState(42)
        lhs_points = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = rng.permutation(n_samples)
            lhs_points[perm, d] = (np.arange(n_samples) + rng.rand(n_samples)) / n_samples * 10 - 5
        
        for p in lhs_points:
            if not time_ok():
                break
            safe_call(p)

        # Phase 2: CMA-ES from best point
        if time_ok() and best_x is not None:
            x, f = cma_es(best_x, best_f, sigma0=0.5, max_iter=200)
            best_x, best_f = x, f

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
