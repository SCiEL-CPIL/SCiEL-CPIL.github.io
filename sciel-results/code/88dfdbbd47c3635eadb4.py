"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive covariance for sharp ridge landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling
        best_x = None
        best_fx = np.inf
        n_init = min(10, budget)
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)

        x = best_x.copy()
        fx = best_fx
        calls_used = n_init if best_x is not None else 1

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2  # number of selected individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 1) + 2 / (dim + 1)) / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        sigma = 0.5  # initial step size
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        C_eigh = np.linalg.eigh(C)
        C_eigval = C_eigh[0]
        C_eigvec = C_eigh[1]
        # Precompute inverse square root of C
        C_eigval_inv_sqrt = 1.0 / np.sqrt(np.maximum(C_eigval, 1e-10))
        C_inv_sqrt = C_eigvec @ np.diag(C_eigval_inv_sqrt) @ C_eigvec.T

        # Track history for restarts
        restarts = 0
        max_restarts = 3
        no_improve_count = 0
        no_improve_threshold = 50  # generations without improvement

        while calls_used < budget and time.time() < deadline:
            # Generate population
            ys = []
            for j in range(lam):
                if calls_used >= budget or time.time() > deadline:
                    break
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(dim)
                y = x + sigma * C_inv_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                ys.append((fy, y))

            if len(ys) < lam:
                # Not enough evaluations, sort what we have
                ys.sort(key=lambda t: t[0])
                lam_actual = len(ys)
                if lam_actual < mu:
                    mu_actual = lam_actual
                else:
                    mu_actual = mu
            else:
                lam_actual = lam
                mu_actual = mu
                ys.sort(key=lambda t: t[0])

            if mu_actual == 0:
                break

            # Update best
            new_best_fx, new_best_x = ys[0]
            if new_best_fx < best_fx:
                best_fx = new_best_fx
                best_x = new_best_x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Compute weighted mean of best mu
            x_new = np.zeros(dim)
            for k in range(mu_actual):
                x_new += weights[k] * ys[k][1]

            # Update step-size control vector
            d_inv = 1.0 / d_sigma
            diff = (C_inv_sqrt @ (x_new - x)) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (dim + 1)) * chi_n
            ps /= np.maximum(np.linalg.norm(ps), 1e-10) if not hsig else 1.0
            sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update evolution path
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x) / sigma

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_hsig * c_c * (1 - c_c)) * C \
                + c_1 * (C + pc @ pc.T) \
                + c_mu * sum(weights[k] * np.outer(ys[k][1] - x, ys[k][1] - x) for k in range(mu_actual))

            # Re-orthogonalize C periodically to maintain symmetry
            C = (C + C.T) / 2

            # Update eigendecomposition
            C_eigh = np.linalg.eigh(C)
            C_eigval = C_eigh[0]
            C_eigvec = C_eigh[1]
            # Ensure eigenvalues are positive
            C_eigval = np.maximum(C_eigval, 1e-10)
            C_eigval_inv_sqrt = 1.0 / np.sqrt(C_eigval)
            C_inv_sqrt = C_eigvec @ np.diag(C_eigval_inv_sqrt) @ C_eigvec.T

            x = x_new

            # Restart if stuck
            if no_improve_count >= no_improve_threshold:
                no_improve_count = 0
                restarts += 1
                if restarts > max_restarts:
                    break
                # Reset CMA-ES state with larger sigma
                sigma = max(sigma * 2, 1.0)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                C = np.eye(dim)
                C_eigh = np.linalg.eigh(C)
                C_eigval = C_eigh[0]
                C_eigvec = C_eigh[1]
                C_eigval_inv_sqrt = 1.0 / np.sqrt(np.maximum(C_eigval, 1e-10))
                C_inv_sqrt = C_eigvec @ np.diag(C_eigval_inv_sqrt) @ C_eigvec.T
                # Move to best known point
                x = best_x.copy()
                fx = best_fx

            # Early termination if sigma is too small
            if sigma < 1e-8:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
