import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES parameters
        lambda_ = 10  # population size
        mu = lambda_ // 2  # number of selected parents
        # Weights: linear, normalized
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu = 2.0 / (mu + 2.0 - 1)  # learning rate for covariance
        c_sigma = (mu + 0.5) / (d + 4 + mu)  # step-size adaptation rate
        c_c = (4 + mu / d) / (d + 4 + 2 * mu)  # path for covariance
        c_1 = 2 / ((d + 1.3) ** 2 + mu)  # rank-one update
        c_mu = min(1 - c_1, 2 * (nu / 2 + 0.5 - 1) / ((d + 2) ** 2 + mu))  # rank-mu update

        # Initial state
        x_mean = np.zeros(d)
        sigma = 2.0  # initial step size (box is [-5,5]^5, so 2 is reasonable)
        B = np.eye(d)
        D = np.ones(d)
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))

        def sample_population():
            """Generate lambda_ samples from N(x_mean, sigma^2 * C)."""
            # Use Cholesky-like decomposition: C = B * diag(D^2) * B^T
            # Sample z ~ N(0, I), then x = x_mean + sigma * B @ (D * z)
            samples = []
            for _ in range(lambda_):
                z = np.random.randn(d)
                x = x_mean + sigma * (B @ (D * z))
                samples.append(x)
            return samples

        # Initial evaluation
        x0 = np.zeros(d)
        f0 = eval_func(x0)
        if eval_count >= budget:
            return best_x, best_f

        # Main CMA-ES loop
        while eval_count < budget and time.time() < deadline:
            if time.time() > deadline:
                break

            # Generate population
            pop = sample_population()

            # Evaluate population
            fitness = np.empty(lambda_)
            for i in range(lambda_):
                if eval_count >= budget or time.time() > deadline:
                    fitness[i] = float('inf')
                    continue
                fitness[i] = eval_func(pop[i])

            if eval_count >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fitness)
            pop_sorted = [pop[i] for i in idx]
            fit_sorted = fitness[idx]

            # Weighted recombination for new mean
            w_sum = np.sum(weights)
            x_new_mean = np.sum([weights[j] * pop_sorted[j] for j in range(mu)], axis=0)

            # Check for convergence: if step is too small relative to sigma
            if np.linalg.norm(x_new_mean - x_mean) < 1e-14:
                break

            # Update path evolution strategies
            C_sqrt_inv = (B.T / D)  # C^{-1/2} = B^{-1} * diag(1/D)
            # Actually: C = B * diag(D^2) * B^T, so C^{-1/2} = B^{-1} * diag(1/D)
            # We store B and D such that C = (B * D) @ (B * D).T
            # So C^{-1/2} = B^{-1} * diag(1/D)
            # For simplicity, use the transformation:
            # z = C^{-1/2} @ (x - x_mean) / sigma
            # We can compute this as: (B.T @ (D * (B^{-1} @ (x - x_mean)))) / sigma
            # But B is orthogonal in standard CMA-ES (from eigendecomp), so B^{-1} = B.T

            # Simpler: compute C^{-1/2} explicitly via eigendecomposition of C
            # C is symmetric positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip to avoid numerical issues
            eigvals = np.clip(eigvals, 1e-20, None)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Path for covariance
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu * nu) * C_sqrt_inv @ (x_new_mean - x_mean) / sigma

            # Step-size adaptation
            norm_pc = np.linalg.norm(pc)
            c_sigma_hat = c_sigma * (1 + 2 * max(0, (norm_pc / chi_n - 1) / (d + 3)))
            sigma *= np.exp(c_sigma_hat * (norm_pc / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # Path for sigma
            z = C_sqrt_inv @ (x_new_mean - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu) * z

            # Update covariance matrix
            # Rank-one update
            delta_hs = (1 - c_c) ** 2 - 1  # not needed, use standard formula
            C = (1 + c_1 * nu * (np.dot(pc, pc) / (d * nu) - 1) * c_c * (2 - c_c)) * C \
                + c_1 * np.outer(pc, pc)

            # Rank-mu update
            diff = np.array([pop_sorted[j] - x_new_mean for j in range(mu)])
            # Transform: C^{-1/2} @ diff
            diff_transformed = diff @ C_sqrt_inv.T  # shape (mu, d)
            rank_mu_term = np.sum([
                weights[j] * np.outer(diff_transformed[j], diff_transformed[j])
                for j in range(mu)
            ], axis=0)
            C += c_mu * rank_mu_term

            # Symmetrize
            C = (C + C.T) / 2.0

            # Eigendecomposition to update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update mean
            x_mean = x_new_mean

            # Check convergence
            if sigma < 1e-10:
                break

            # Check if we've made minimal progress
            if np.all(fit_sorted[:mu] - fit_sorted[-1] < 1e-10) and sigma < 1e-6:
                break

        return best_x, best_f
