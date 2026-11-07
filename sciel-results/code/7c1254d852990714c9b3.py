"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Step size
        sigma = 1.0
        # Initial mean: center of the box
        x_mean = np.zeros(dim)
        fx_mean = func(x_mean)
        best_x = x_mean.copy()
        best_f = fx_mean
        call_count = 1

        # Covariance matrix (start as identity)
        C = np.eye(dim)
        # Eigen decomposition for updating C
        # We'll use a simplified update to avoid full eigendecomp each step for speed
        # Actually, for dim=5, full eigendecomp is cheap.

        # Learning rates
        mu = lam // 2  # number of selected solutions
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Initial evaluation of population
        # Generate first population
        rng = np.random.default_rng(42)

        def evaluate_population():
            nonlocal call_count, best_x, best_f
            # Generate lam samples
            # Use Cholesky of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigendecomp if C is not positive definite
                w, V = np.linalg.eigh(C)
                w = np.maximum(w, 1e-16)
                L = V @ np.diag(np.sqrt(w)) @ V.T
                # Actually, for sampling we want sqrt(C), so V @ diag(sqrt(w))
                L = V @ np.diag(np.sqrt(w))
            z = rng.standard_normal((lam, dim))
            # Samples: x_mean + sigma * z @ L.T (since C = L L^T, samples are x_mean + sigma * (z @ L^T))
            # Wait, if C = L L^T, then a random vector with covariance C is L @ u where u ~ N(0, I).
            # So z @ L.T is equivalent to (L @ z.T).T = L @ z.T, which is a row vector.
            # Let's use: samples = x_mean + sigma * (z @ L.T)
            samples = x_mean + sigma * (z @ L.T)
            # Clip to box [-5, 5]
            samples = np.clip(samples, -5, 5)

            # Evaluate
            fvals = np.array([func(s) for s in samples])
            call_count += lam

            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = samples[idx_min].copy()

            return samples, fvals

        # Initial population
        samples, fvals = evaluate_population()

        # Sort by fitness
        order = np.argsort(fvals)
        samples = samples[order]
        fvals = fvals[order]

        # Main loop
        while call_count < budget:
            # Check time
            if time.time() > deadline:
                break

            # Update mean
            x_new = np.sum(weights[:, None] * samples[:mu], axis=0)

            # Update step size
            # p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mueff) * (C^(-1/2) * (x_new - x_mean) / sigma)
            # C^(-1/2) is expensive. Approximate with inverse of L?
            # Let's use the approximation: C^(-1/2) @ v ≈ L^{-T} @ L^{-1} @ v ? No.
            # Let's stick to a simpler CMA-ES variant or just use (1+1)-ES with better restarts?
            # Given dim=5, let's try a simpler covariance adaptation or just use (1+1)-ES with larger sigma exploration.
            # Actually, the seed (1+1)-ES scored 1.0. CMA-ES is known to be better for rotated Rastrigin.
            # Let's implement a lightweight CMA-ES.

            # For speed, let's use a simplified update:
            # p_sigma update requires C^{-1/2}. Let's approximate C^{-1/2} using the Cholesky factor L.
            # v = x_new - x_mean
            # u = L^{-1} @ v  (solves L u = v)
            # w = L^{-T} @ u (solves L^T w = u)
            # Then C^{-1/2} v ≈ w (if C = L L^T, C^{-1/2} = L^{-T} L^{-1})
            v = x_new - x_mean
            try:
                u = np.linalg.solve(L, v)
                w = np.linalg.solve(L.T, u)
            except np.linalg.LinAlgError:
                w = v / np.linalg.norm(v) * np.sqrt(dim) if np.linalg.norm(v) > 0 else np.zeros(dim)

            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * w

            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (call_count + 1)))) < (1.4 + 2 / (dim + 1)) * chiN
            hsig = float(hsig)

            # Update p_c
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * (v / sigma)

            # Update C
            # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (1 - hsig) * cc * (2 - cc) * C) + cmu * sum(weights_i * (y_i - x_old)(y_i - x_old)^T)
            # where y_i = (samples[i] - x_mean) / sigma
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            rank_one = p_c @ p_c.T
            rank_mu = np.sum(weights[:, None, None] * (samples[:mu] - x_mean)[:, None, :] / sigma * (samples[:mu] - x_mean)[None, :, :] / sigma, axis=0)
            C = (1 - c1 - cmu) * C + c1 * (rank_one + delta_hsig * C) + cmu * rank_mu

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chiN - 1))

            # Update mean
            x_mean = x_new
            # Evaluate new mean? No, CMA-ES generates new population from new mean.
            # But we need to keep track of best.
            # The best found so far is best_x, best_f.

            # Generate next population
            samples, fvals = evaluate_population()
            order = np.argsort(fvals)
            samples = samples[order]
            fvals = fvals[order]

        return best_x, best_f
# EVOLVE-BLOCK-END
