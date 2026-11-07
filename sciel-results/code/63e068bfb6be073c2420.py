"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.full(dim, 5.0)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 2.0

        rng = np.random.default_rng(42)

        # CMA-ES implementation
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))
        mu = lambda_pop // 2  # number of selected individuals
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Eigen decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        # Step size
        sigma = 2.0
        # Mean
        x_mean = np.zeros(dim)
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Initial evaluation
        fx = func(x_mean)
        calls += 1
        update_best(x_mean, fx)

        # Main CMA-ES loop
        while calls < budget and time_left():
            # Generate population
            Y = []
            for i in range(lambda_pop):
                # Compute inverse square root of C using cached eigen decomposition
                C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                z = rng.standard_normal(dim)
                y = C_inv_sqrt @ z
                Y.append(y)
            Y = np.array(Y)
            
            # Transform to x-space
            X = np.array([x_mean + sigma * y for y in Y])
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lambda_pop
            for x, f in zip(X, F):
                update_best(x, f)
            
            # Sort by fitness
            idx = np.argsort(F)
            Y_selected = Y[idx[:mu]]
            F_selected = F[idx[:mu]]
            
            # Update mean
            x_mean_new = x_mean + sigma * (w @ Y_selected)
            
            # Update step size
            # Need C_inv_sqrt for this
            C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            norm_y = np.linalg.norm(C_inv_sqrt @ (x_mean_new - x_mean)) / sigma if sigma > 0 else 0
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (x_mean_new - x_mean) / sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_pop))) < (1 + 2 / (dim + 1)) * np.trace(C_inv_sqrt @ C_inv_sqrt) / dim else 0
            sigma *= np.exp(cs / dim * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_pop))) - 1))
            
            # Update covariance
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (x_mean_new - x_mean) / sigma
            delta_h = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - cc) ** (2 * (calls // lambda_pop))) < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 2 / (dim + 1)) else 0
            C = (1 + c1 * delta_h - c1) * C + c1 * np.outer(p_c, p_c) + cmu * (w @ np.outer(Y_selected, Y_selected))
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
            except:
                C = np.eye(dim)
                eigvals = np.ones(dim)
                eigvecs = np.eye(dim)
            
            x_mean = x_mean_new
        
        return best_x, best_f

# EVOLVE-BLOCK-END
