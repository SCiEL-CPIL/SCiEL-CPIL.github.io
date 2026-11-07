"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start CMA-ES with progressively smaller initial step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track the best solution found so far
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start with progressively smaller initial step sizes
        # For Sphere function, the global optimum is unique, so we can
        # use deterministic starting points with different scales
        initial_scales = [3.0, 1.0, 0.3]
        # Allocate budget across restarts: first gets most, subsequent get less
        budget_per_restart = [int(budget * 0.5), int(budget * 0.35), int(budget * 0.15)]

        for start_idx, sigma0 in enumerate(initial_scales):
            if calls_made >= budget:
                break
            
            remaining = budget - calls_made
            alloc = min(budget_per_restart[start_idx], remaining)
            if alloc < 10:  # Need minimum population evaluations
                break

            # Start from center
            x = np.zeros(dim)
            fx = evaluate(x)

            # CMA-ES parameters
            mu = max(2, dim // 2)  # number of parents (at least 2)
            lambda_ = max(4, 2 * mu)  # population size (at least 4)
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            
            # Adaptation rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            # Initialize CMA-ES state
            sigma = sigma0
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            
            # Precompute inverse square root of C
            C_inv_sqrt = np.eye(dim)

            # Stagnation detection
            best_fx_in_restart = fx
            no_improve_count = 0
            stagnation_threshold = max(15, int(alloc * 0.12))

            for iteration in range(alloc):
                if calls_made >= budget:
                    break

                # Sample population
                Y = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    Y[i] = x + sigma * (C_inv_sqrt @ rng.standard_normal(dim))
                
                # Evaluate
                F = np.array([evaluate(Y[i]) for i in range(lambda_)])
                
                # Sort by fitness
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]

                # Update best
                if F_sorted[0] < best_fx_in_restart - 1e-10:
                    best_fx_in_restart = F_sorted[0]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Mean update
                x_old = x.copy()
                x = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
                
                # Strategy parameter update
                y_sigma = (x - x_old) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                
                # Hessian of the path (for h_sigma computation)
                chi_n = np.sqrt(dim) * (1 + 1 / (4 * dim) - 1 / (21 * dim ** 2))
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))
                           < (1.4 + 2 / (dim + 1)) * chi_n)
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma

                # Covariance matrix update (corrected formula)
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                
                # Rank-mu update
                diff = Y_sorted[:mu] - x_old
                rank_mu = np.sum(w[:, None, None] * np.outer(diff, diff), axis=0)
                
                C = ((1 + c_1 * delta_h - c_1 - c_mu) * C
                     + c_1 * rank_one
                     + c_mu * rank_mu)
                
                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-12):
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update inverse square root
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Sigma update
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Check for stagnation
                if no_improve_count >= stagnation_threshold:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
