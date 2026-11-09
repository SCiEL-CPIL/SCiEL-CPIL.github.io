"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin) with CMA-ES."""
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d))  # ~9 for d=5
        # Initial sigma
        sigma = 1.0
        # Initial mean
        x = np.zeros(d)
        # Initial covariance
        C = np.eye(d)
        # Eigen decomposition cache
        D_inv_sqrt = np.ones(d)
        B = np.eye(d)
        # Learning rates
        mu = lam // 2  # number of selected points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # State
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        best_x = x.copy()
        best_f = np.inf

        def evaluate(x_vec):
            nonlocal best_x, best_f
            x_clipped = np.clip(x_vec, -5.0, 5.0)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Initial evaluation
        evaluate(x)
        calls_made = 1

        # Eigen decomposition function
        def eig(C_mat):
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvals, eigvecs

        # Main loop
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Eigen decomposition
            eigvals, eigvecs = eig(C)
            D_inv_sqrt = 1.0 / np.sqrt(eigvals)
            B = eigvecs

            # Generate population
            y = np.zeros((lam, d))
            for i in range(lam):
                z = rng.standard_normal(d)
                y[i] = B @ (D_inv_sqrt * z)

            # Generate candidates
            candidates = []
            for i in range(lam):
                xi = x + sigma * y[i]
                candidates.append((xi, i))

            # Evaluate candidates (with time check)
            evals = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                xi = candidates[i][0]
                fi = evaluate(xi)
                evals.append((fi, i))
                calls_made += 1

            if len(evals) < mu:
                break

            # Sort by fitness
            evals.sort(key=lambda t: t[0])

            # Weighted recombination
            y_w = np.zeros(d)
            for k in range(mu):
                y_w += weights[k] * y[evals[k][1]]

            # Update mean
            x = x + sigma * y_w

            # Update step-size control
            y_old = B @ (D_inv_sqrt * y_w)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_old

            # Update CMA (with h_sigma check)
            h_sigma = (np.linalg.norm(p_sigma) / (1 - (1 - c_sigma) ** (calls_made // lam + 1)) / chi_n) < 1.4 + 2 / (d + 1)

            # Update evolution path
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_old

            # Rank-one update
            delta_h = (1 - h_sigma) * np.sqrt(c_c * (2 - c_c) * mu_eff)
            C = (1 + c_1 * (2 - c_1) + c_mu) * C - c_1 * np.outer(p_c, p_c) - delta_h * np.outer(p_c, p_c) + c_mu * np.outer(y_w, y_w)

            # Rank-mu update
            for k in range(mu):
                C += c_mu * weights[k] * np.outer(y[evals[k][1]], y[evals[k][1]])

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Clamp x to bounds
            x = np.clip(x, -5.0, 5.0)

        return best_x, best_f
