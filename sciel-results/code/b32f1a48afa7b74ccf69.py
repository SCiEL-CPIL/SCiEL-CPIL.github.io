"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    - Random sampling phase; covariance initialized from top samples (PCA)
      to align with the funnel structure quickly.
    - Full CMA-ES with covariance matrix adaptation and restarts.
    - Final Nelder-Mead polish for tight local refinement.
    - Time guard to stay within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # ---- Phase 1: Random sampling ----
        n_samples = min(60, budget // 8)
        samples = []
        best_x = None
        best_fx = float('inf')

        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            samples.append((fx, x.copy()))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        calls_used = n_samples

        # ---- Initialize covariance from top samples (PCA) ----
        samples_sorted = sorted(samples, key=lambda t: t[0])
        n_top = max(5, min(15, len(samples_sorted) // 3))
        top_samples = np.array([s[1] for s in samples_sorted[:n_top]])

        if n_top > 1:
            cov = np.cov(top_samples, rowvar=False)
            if cov.size == 0:
                cov = np.eye(dim)
            # Regularize
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            # Scale to a reasonable magnitude
            scale = np.sqrt(np.mean(np.diag(cov))) if np.trace(cov) > 0 else 1.0
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        else:
            cov = np.eye(dim)

        # ---- Phase 2: CMA-ES ----
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        m = best_x.copy()
        sigma = 1.0
        C = cov.copy()
        try:
            C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            C_sqrt = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        stall_count = 0
        max_stall = 15
        restart_sigma_mult = 1.5

        while calls_used < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            evaluated = []
            for y in offspring:
                if calls_used >= budget or time.time() > deadline - 5.0:
                    break
                fy = func(y)
                calls_used += 1
                evaluated.append((fy, y))

            if len(evaluated) < pop_size:
                break

            evaluated.sort(key=lambda t: t[0])

            if evaluated[0][0] < best_fx:
                best_fx = evaluated[0][0]
                best_x = evaluated[0][1].copy()
                stall_count = 0
            else:
                stall_count += 1

            # Check for stagnation
            if stall_count >= max_stall:
                # Restart with larger sigma and re-center
                sigma *= restart_sigma_mult
                stall_count = 0
                # Random re-centering
                m = best_x + rng.standard_normal(dim) * 0.5
                m = np.clip(m, -5, 5)
                # Reset covariance to identity
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                continue

            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (evaluated[i][1] - m) / sigma
            m = m + sigma * y_w
            m = np.clip(m, -5, 5)

            # Update step size
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_sigma += weights[i] * (evaluated[i][1] - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            p_sigma_norm = np.linalg.norm(p_sigma)
            h_sigma = (p_sigma_norm / chi_n < (1.4 + 2 / (dim + 1)) * np.sqrt(1 + (1 - c_sigma) ** (2 * calls_used / (pop_size * (dim + 1)))))
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_n - 1))

            # Update covariance
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (evaluated[i][1] - m) / sigma
            p_c = (1 - c1) * p_c + h_sigma * np.sqrt(c1 * (2 - c1) * mu_eff) * y_w

            C = (1 + c1 * (h_sigma * (2 - c1) - 1) + cmu) * C
            for i in range(mu):
                y_i = (evaluated[i][1] - m) / sigma
                C -= cmu * weights[i] * (np.outer(y_i, y_i) - C)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

        # ---- Phase 3: Local polish with Nelder-Mead ----
        if calls_used < budget and time.time() < deadline - 2.0:
            # Use a simple local search to polish
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            for _ in range(min(50, budget - calls_used)):
                if time.time() > deadline - 2.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-4, 1.0)

        return best_fx
# EVOLVE-BLOCK-END
