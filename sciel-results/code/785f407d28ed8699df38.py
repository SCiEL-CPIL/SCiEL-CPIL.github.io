"""CMA-ES based optimizer for BBOB-style bent cigar function f12."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES Optimizer for anisotropic landscape adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 320.0  # 330s kill, 10s margin

        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(np.asarray(x, dtype=np.float64))

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = int(4 + 3 * np.log(dim))  # 7 for dim=5
        # Step size
        sigma = 2.0  # initial sigma, box is [-5,5] so 2 is reasonable

        # Mean
        mean = np.zeros(dim)

        # Covariance matrix (initially identity)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Cholesky decomposition
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Evolution path weights
        mu = lam // 2  # number of parents
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w ** 2)

        # Adaptation rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Step size control
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Best tracking
        best_x = np.zeros(dim)
        best_fx = float('inf')

        # Initial evaluation
        best_fx = eval_func(mean)
        best_x = mean.copy()

        # Generation loop
        while self.calls < budget and (time.time() - self.start_time) < self.time_limit:
            # Generate population
            population = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = C_sqrt @ z
                x = mean + sigma * y
                population.append((x, y, z))

            # Evaluate
            f_values = []
            for i in range(lam):
                x, y, z = population[i]
                fx = eval_func(x)
                f_values.append(fx)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

            # Sort by fitness
            indices = np.argsort(f_values)
            sorted_y = np.array([population[i][1] for i in indices])
            sorted_f = [f_values[i] for i in indices]

            # Update mean (recombination)
            mean_old = mean.copy()
            mean = sum(w[i] * sorted_y[i] for i in range(mu))
            mean = mean_old + sigma * mean

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_sqrt_inv @ (mean - mean_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (mean - mean_old) / sigma

            # Hessian correction
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (self.calls // lam + 1)))) / chi_n
            h_sig = min(1.0, h_sig)

            # Update covariance
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            delta_h = 1 + delta_h

            rank1 = np.outer(p_c, p_c)
            rankmu = sum(w[i] * np.outer(sorted_y[i], sorted_y[i]) for i in range(mu))

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank1 + delta_h * np.eye(dim)) + c_mu * rankmu

            # Ensure positive definite
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                # Regularize
                C += 1e-8 * np.eye(dim)
                try:
                    C_inv = np.linalg.inv(C)
                    C_sqrt = np.linalg.cholesky(C)
                    C_sqrt_inv = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_sqrt_inv = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(1e-12, min(sigma, 1e3))

            # Stagnation detection: if sigma is too small or no improvement for many generations
            # Simple check: if best_fx hasn't improved significantly in last 50 evaluations
            if self.calls > 50:
                # Check if we're stuck
                if sigma < 1e-10:
                    # Restart with larger sigma around best point
                    sigma = 0.5
                    mean = best_x.copy()
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_sqrt_inv = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)

        # Final polish with local search if we have budget remaining
        if self.calls < budget and (time.time() - self.start_time) < self.time_limit - 10:
            # Nelder-Mead polish
            try:
                from scipy.optimize import minimize
                res = minimize(
                    eval_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except (StopIteration, Exception):
                pass

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(dim)
            if self.calls < budget:
                best_fx = eval_func(best_x)
            else:
                best_fx = 0.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
