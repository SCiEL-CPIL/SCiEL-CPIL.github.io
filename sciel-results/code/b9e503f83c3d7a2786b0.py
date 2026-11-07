import numpy as np
import time

class Optimizer:
    """
    CMA-ES with multi-start initialization and adaptive sigma for BBOB f13 (Sharp ridge).
    Combines robust covariance adaptation for anisotropic ridges with a simple
    success-rate based sigma adaptation for early progress.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_evals = 0

        def eval_point(x):
            nonlocal total_evals, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            total_evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---------- Phase 1: Multi-start Initialization ----------
        # Sample random points to locate the basin of the sharp ridge.
        n_starts = 20
        for _ in range(n_starts):
            if total_evals >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # ---------- Phase 2: CMA-ES Refinement ----------
        # CMA-ES adapts the covariance matrix to the local landscape,
        # which is crucial for sharp ridges with anisotropic curvature.
        
        x_mean = best_x.copy()
        
        # CMA-ES parameters
        lambda_ = 5  # Small population to stay on the ridge and allow more generations
        sigma = 1.0
        
        # Covariance matrix C and its Cholesky factor T
        C = np.eye(dim)
        
        def get_transformation(C):
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            return L

        T = get_transformation(C)

        # Weights for recombination
        weights = np.array([np.log(lambda_ / 2 + 0.5) - np.log(i + 1) for i in range(lambda_)])
        weights /= np.sum(weights)
        mu = lambda_ // 2  # Number of parents
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        
        # Adaptation rates
        c_sigma = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 2) / (dim + 1) - 1) + c_sigma)
        cs = (dim + lambda_ + 2) / (dim + 5 + 3 * lambda_)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2) + 1) / ((dim + 2) ** 2 + mu_eff))

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Main CMA-ES loop
        while total_evals < budget and time.time() < deadline:
            # Generate population
            y = np.zeros((lambda_, dim))
            z = np.zeros((lambda_, dim))
            f_vals = np.zeros(lambda_)
            
            pop_complete = True
            for i in range(lambda_):
                if total_evals >= budget or time.time() > deadline:
                    pop_complete = False
                    break
                z[i] = rng.standard_normal(dim)
                y[i] = x_mean + sigma * (T @ z[i])
                f_vals[i] = eval_point(y[i])
                y[i] = np.clip(y[i], -5.0, 5.0)  # Update y with clipped value for mean update

            if not pop_complete:
                break

            # Sort by fitness
            order = np.argsort(f_vals)
            y_sorted = y[order]
            f_sorted = f_vals[order]
            z_sorted = z[order]
            
            # Update mean
            x_mean_new = np.sum(weights[:mu, np.newaxis] * y_sorted[:mu], axis=0)
            
            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                T.T @ (weights[:mu, np.newaxis] * z_sorted[:mu]).sum(axis=0)
            )
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (total_evals // lambda_))) < \
                        (1 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 1) else 0
            
            pc = (1 - cs) * pc + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (
                T.T @ (weights[:mu, np.newaxis] * z_sorted[:mu]).sum(axis=0)
            )
            
            # Update covariance
            delta_h = (h_sigma - 1) * cs * (2 - cs)
            C = (1 + c1 * delta_h - c1 - cmu) * C + \
                c1 * (np.outer(ps, ps) + (1 - h_sigma) * delta_h * C) + \
                cmu * np.sum(weights[:mu, np.newaxis, np.newaxis] * np.outer(y_sorted[:mu] - x_mean, y_sorted[:mu] - x_mean), axis=0)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update transformation matrix
            T = get_transformation(C)
            
            # Update best
            x_mean = x_mean_new

        # ---------- Phase 3: Polishing ----------
        # Final local refinement around the best point found by CMA-ES.
        if total_evals < budget and time.time() < deadline:
            polish_sigma = max(sigma * 0.5, 0.05)
            polish_budget = min(budget - total_evals, 50)
            for i in range(polish_budget):
                if time.time() > deadline or total_evals >= budget:
                    break
                y = best_x + polish_sigma * rng.standard_normal(dim)
                fy = eval_point(y)
                polish_sigma *= 0.95

        return best_x, best_f
