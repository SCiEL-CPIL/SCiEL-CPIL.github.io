"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage optimizer: LHS bracketing + CMA-ES with efficient sampling."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 310.0  # safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf

        rng = np.random.default_rng(12345)

        def time_ok():
            return time.time() < deadline

        # Stage 1: Coarse Latin Hypercube + center + a few random
        # 3-level LHS in 5 dims: 3^5 = 243, but we only have ~20 evals
        # Use a stratified sampling with 4 points per dim axis
        init_budget = 20
        candidates = []
        # Center
        candidates.append(np.zeros(dim))
        # 3-level LHS: for each dim, split [-4,4] into 3 strata
        # Use a simple stratified approach: 3^dim would be too many, so use random stratified
        # Actually with 20 evals, let's do a mixed approach:
        # 1 center + 10 random stratified + 9 random
        for d in range(dim):
            # 2 points per axis (positive and negative)
            candidates.append(np.full(dim, 0.0) + np.eye(dim)[d] * 2.0)
            candidates.append(np.full(dim, 0.0) - np.eye(dim)[d] * 2.0)
        # That's 1 + 10 = 11 candidates
        # Add more random points
        while len(candidates) < init_budget:
            candidates.append(rng.uniform(-4, 4, dim))

        # Evaluate all
        used = 0
        cand_vals = []
        for c in candidates:
            if not time_ok() or used >= total_budget:
                break
            c_clipped = np.clip(c, -5, 5)
            v = func(c_clipped)
            cand_vals.append((v, c_clipped))
            used += 1
            if v < best_f:
                best_f = v
                best_x = c_clipped.copy()

        if not cand_vals:
            return best_x, best_f

        # Sort by fitness
        cand_vals.sort(key=lambda x: x[0])
        x = cand_vals[0][1].copy()
        fx = cand_vals[0][0]

        # Stage 2: CMA-ES
        remaining = total_budget - used
        if remaining <= 0 or not time_ok():
            return best_x, best_f

        # CMA-ES parameters
        lam = 10  # smaller population for efficiency
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Small initial sigma since we're starting from a good bracket
        sigma = 0.5
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute Cholesky for sampling
        def sample_from_C(C, rng, n_samples):
            """Sample n_samples points from N(0, C) using Cholesky."""
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Fallback to eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            Z = rng.standard_normal((n_samples, dim))
            return Z @ L.T

        cma_left = remaining - 30  # reserve 30 for polish
        max_iters = max(1, cma_left // lam)

        for it in range(max_iters):
            if not time_ok() or used >= total_budget:
                break

            # Sample lambda points
            Y = sample_from_C(C, rng, lam)
            X = np.clip(x[None, :] + sigma * Y, -5, 5)

            # Evaluate
            F = np.array([func(xi) for xi in X])
            used += lam

            # Update best
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            # Sort by fitness
            order = np.argsort(F)
            F_sorted = F[order]
            Y_sorted = Y[order]
            X_sorted = X[order]

            # Mean update
            x_old = x.copy()
            x = x_old + sigma * np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            x = np.clip(x, -5, 5)

            # Covariance matrix adaptation
            ys = Y_sorted[:mu]
            mean_y = np.sum(w[:, None] * ys, axis=0)

            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_y

            # Update C
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += w[i] * np.outer(ys[i], ys[i])

            coeff_c = (1 - c_c) * (1 - c_sigma)
            coeff_pc = c_c * (1 - c_c) * d_sigma
            coeff_h = c_sigma * (1 - 1.0 / mu_eff)

            C_new = coeff_c * C + coeff_pc * np.outer(p_c, p_c) + coeff_h * H
            C_new = (C_new + C_new.T) / 2

            # Ensure PSD
            eigvals_C, eigvecs_C = np.linalg.eigh(C_new)
            if np.min(eigvals_C) < 1e-12:
                C_new = eigvecs_C @ np.diag(np.clip(eigvals_C, 1e-10, None)) @ eigvecs_C.T
            C = C_new

            # Step-size control
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * mean_y
            sigma *= np.exp((1.0 / (5 * dim + 5)) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Termination: if sigma is very small, break
            if sigma < 1e-8:
                break

        # Stage 3: (1+1)-ES polish
        if time_ok() and used < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = max(0.05, sigma * 0.2)
            polish_left = total_budget - used
            for _ in range(polish_left):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.75
                    if sigma < 1e-10:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
