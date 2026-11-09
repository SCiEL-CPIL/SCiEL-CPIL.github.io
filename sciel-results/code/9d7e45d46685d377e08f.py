import numpy as np
import time
import scipy.stats.qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0
        rng = np.random.default_rng(42)

        # Phase 1: Small global scan with Sobol points
        n_scan = 20
        sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(n_scan)
        points = points * 10 - 5

        best_x = None
        best_f = np.inf
        all_candidates = []

        for x in points:
            if time.time() > deadline - 10.0:
                break
            fx = func(x)
            all_candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x0 = rng.uniform(-5, 5, self.dim)
            return x0, func(x0)

        # Phase 2: Covariance Matrix Adaptation (CMA-ES-like) from the best point
        # This is a structural change: instead of a simple random-walk ES with
        # sigma adaptation, we use an adaptation of the full covariance matrix,
        # which better captures the local landscape geometry.
        remaining = self.budget - len(all_candidates)
        if remaining <= 0:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = 20
        # Adaptation rates
        c_sigma = (pop_size + 1.5) / (self.dim + pop_size + 5)
        c_c = (4 + pop_size / self.dim) / (self.dim + 4 + 2 * pop_size / self.dim)
        mu = pop_size // 2
        # Weights: linearly decreasing, normalized
        w = np.arange(mu, 0, -1, dtype=float)
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        x_best_local = best_x.copy()
        f_best_local = best_f
        sigma = 1.0
        pc = np.zeros(self.dim)
        cs = np.zeros(self.dim)
        C = np.eye(self.dim)
        # Precompute C^{1/2} initially
        C_sqrt_inv = np.eye(self.dim)

        for iteration in range(remaining // pop_size + 1):
            if time.time() > deadline - 10.0:
                break

            # Generate population
            pop = []
            for i in range(pop_size):
                # Use Cholesky factor for C
                if i == 0:
                    # Recompute sqrt of C every generation for stability
                    try:
                        C_cholesky = np.linalg.cholesky(C)
                        C_sqrt_inv = np.linalg.inv(C_cholesky)
                    except np.linalg.LinAlgError:
                        C_cholesky = np.eye(self.dim)
                        C_sqrt_inv = np.eye(self.dim)

                z = rng.standard_normal(self.dim)
                d = C_cholesky @ z
                y = x_best_local + sigma * d
                y = np.clip(y, -5, 5)
                pop.append((y, func(y)))

            # Sort by fitness
            pop.sort(key=lambda t: t[1])

            # Update best
            if pop[0][1] < f_best_local:
                f_best_local = pop[0][1]
                x_best_local = pop[0][0].copy()
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()

            # Weighted mean
            m_new = np.zeros(self.dim)
            for i in range(mu):
                m_new += w[i] * pop[i][0]

            # Update step size (sigma)
            # Using the standard CMA-ES sigma update
            diff = m_new - x_best_local
            # Approximate ||C^{1/2}^{-1} diff||
            C_sqrt_inv_diff = C_sqrt_inv @ diff
            norm = np.linalg.norm(C_sqrt_inv_diff)
            if norm > 1e-30:
                c_sigma_norm = c_sigma / (1 + c_sigma)
                # h_sigma = 1 if norm < (1 + c_sigma) * (1 + sqrt((mu_eff-2)/(dim+2)))
                # Simplified: use expected norm of N(0,C) which is sqrt(dim)
                # For stability, just use a simple adaptation
                expected_norm = np.sqrt(np.trace(C)) * np.sqrt(1.0 / (self.dim))
                # Actually, let's use the standard formula more carefully
                # The standard CMA-ES uses:
                # sigma = sigma * exp(c_sigma / (1 + c_sigma) * (||C^{-1/2}(m - x)|| / E||N(0,C)|| - 1))
                # This is getting complex. Let's use a simpler but robust approach:
                # Adapt sigma based on the success rate of the top mu points
                success_count = sum(1 for i in range(mu) if pop[i][1] < f_best_local)
                success_rate = success_count / mu
                if success_rate > 0.5:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.8
                else:
                    sigma *= 0.99
                sigma = np.clip(sigma, 0.001, 5.0)

            # Update path aggregate for mean (pc)
            pc = (1 - c_sigma) * pc + c_sigma * (m_new - x_best_local) / sigma

            # Update path aggregate for covariance (cs)
            # Simplified: use the difference between m_new and x_best_local
            # For a more standard CMA-ES, we'd use the standard normal transform
            # But for robustness, let's use a simplified rank-mu update
            cs = (1 - c_c) * cs + c_c * (m_new - x_best_local) / sigma

            # Update covariance matrix
            # Rank-mu update
            delta_C = np.zeros((self.dim, self.dim))
            for i in range(mu):
                diff_i = (pop[i][0] - x_best_local) / sigma
                delta_C += w[i] * np.outer(diff_i, diff_i)

            # Rank-1 update for the path
            delta_C += (1 + (2 - mu_eff / mu) * c_c * (1 - c_c)) * np.outer(cs, cs)

            C = (1 - (c_sigma + mu_eff / 2) * c_sigma + mu_eff * c_sigma) * C + delta_C

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_best_local = m_new

        # Phase 3: Final polish with very small steps
        polish_budget = 30
        if time.time() < deadline - 5.0:
            polish_sigma = 0.05
            for _ in range(polish_budget):
                if time.time() > deadline - 2.0:
                    break
                y = np.clip(best_x + polish_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x = y
                    best_f = fy
                polish_sigma *= 0.95

        return best_x, best_f
