import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget
        calls_left = [budget]

        def guarded(x):
            if calls_left[0] <= 0:
                return float('inf')
            calls_left[0] -= 1
            try:
                return float(func(x))
            except Exception:
                return float('inf')

        # Phase 1: Sobol scan for good initialization
        n_scan = min(60, max(10, budget // 8))
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            u = sampler.random(n_scan)
        except Exception:
            rng = np.random.default_rng(42)
            u = rng.random((n_scan, dim))
        
        pts = -5.0 + 10.0 * u
        pts = np.clip(pts, -5, 5)
        
        best_val = float('inf')
        best_x = None
        
        for i in range(n_scan):
            if time.time() >= deadline or calls_left[0] <= 0:
                break
            x = pts[i]
            fx = guarded(x)
            if fx < best_val:
                best_val = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_val = guarded(best_x)

        # Phase 2: CMA-ES style optimization with adaptive covariance
        x = best_x.copy()
        fx = best_val
        
        # Initialize CMA-ES parameters
        sigma = 2.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu = max(1, dim // 2 + 1)  # Number of parents
        lambda_ = 2 * mu  # Population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Cache for inverse square root of C
        def update_cma_matrices():
            nonlocal C, C_inv, C_sqrt, C_sqrt_inv
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to avoid numerical issues
                eigvals = np.clip(eigvals, 1e-20, None)
                C_inv = (eigvecs * (1.0 / eigvals)) @ eigvecs.T
                C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            except Exception:
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
        
        update_cma_matrices()
        
        # Main CMA-ES loop
        while calls_left[0] > lambda_ and time.time() < deadline:
            # Generate population
            y_offspring = []
            for i in range(lambda_):
                if time.time() >= deadline or calls_left[0] <= 0:
                    break
                z = np.random.randn(dim)
                y = C_sqrt @ z
                y_offspring.append(y)
            
            if len(y_offspring) < mu:
                break
            
            # Evaluate offspring
            offspring = []
            for y in y_offspring[:lambda_]:
                x_trial = x + sigma * y
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = guarded(x_trial)
                offspring.append((f_trial, x_trial, y))
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Selection
            selected = offspring[:mu]
            new_x = sum(weights[i] * selected[i][1] for i in range(mu))
            
            # Update evolution paths
            y_sigma = sum(weights[i] * selected[i][2] for i in range(mu))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            
            # Update mean
            x = new_x
            fx = selected[0][0]
            
            # Update covariance
            delta_h = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (lambda_ + 1)))) - (1 - c_sigma) * chi_n
            h_sig = 1.0 if delta_h < 1 / c_sigma else 0.0
            
            outer = np.outer(p_c, p_c)
            C = (1 + c_1 * h_sig * (dim + 1) - c_1 - c_mu) * C \
                + c_1 * (outer - C) \
                + c_mu * sum(weights[i] * np.outer(selected[i][2], selected[i][2]) for i in range(mu))
            
            # Ensure symmetry
            C = 0.5 * (C + C.T)
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Update CMA matrices periodically
            if calls_left[0] % lambda_ == 0:
                update_cma_matrices()
        
        return x, fx
