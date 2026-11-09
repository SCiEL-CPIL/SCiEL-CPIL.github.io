"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation) for the Bent Cigar BBOB f12."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 300.0 - 5.0  # safety margin before the 300 s kill
        n, dim = self.budget, self.dim

        rng = np.random.default_rng(0)

        # ---- CMA-ES state ----
        # Start near center; the hidden shift lies in [-4,4]^5, center is a safe
        # and well-conditioned starting point for the Bent Cigar.
        x = np.zeros(dim)
        fx = func(x)
        best_x, best_f = x.copy(), fx

        # Initial step size: a fraction of the box half-width.
        sigma = 4.0
        # Population size for CMA-ES in dim=5.
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        # Weighted recombination weights (rank-1 / rank-mu adaptation).
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1.5) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   (2 + (mu_eff / dim - 1) ** 2) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        # Precompute Cholesky of C (identity initially).
        C_chol = np.linalg.cholesky(C)

        # Keep a small local history of best points for a final polish pass.
        history = []

        def _clip(v):
            return np.clip(v, -5.0, 5.0)

        calls = 1
        for it in range(1, n):
            # Time guard: stop early and return best if near deadline.
            if time.time() > deadline:
                break

            # ---- Sample population ----
            # Move mean by the evolution path.
            x = _clip(x)
            # Sample lambda candidates.
            ys = []
            for k in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (C_chol @ z)
                ys.append(_clip(y))
            ys = np.array(ys)
            calls += lam
            if calls > n:
                break

            # Evaluate.
            fvals = np.array([func(y) for y in ys])
            # Update best.
            idx_best = int(np.argmin(fvals))
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = ys[idx_best].copy()

            # Sort by fitness.
            order = np.argsort(fvals)
            ys_sorted = ys[order]
            fvals_sorted = fvals[order]
            # Track history of best few for polish.
            for kk in range(min(2, lam)):
                history.append(ys_sorted[kk].copy())
            if len(history) > 200:
                history.pop(0)

            # ---- Weighted recombination ----
            y_mean = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            d = y_mean - x
            # Evolution paths.
            inv_chol = np.linalg.inv(C_chol)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (inv_chol @ d)
            cs = np.linalg.norm(pc) / chi_n
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_chol @ d / sigma)
            # Adapt sigma.
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-12, 50.0))

            # ---- Covariance adaptation ----
            delta = (1 if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (dim + 1)) else 0)
            C = ((1 - c_1 - c_mu) * C
                 + c_1 * ((pc @ pc.T) + (1 - delta) * c_c * (2 - c_c) * C)
                 + c_mu * np.sum(weights[:, None, None] * (ys_sorted[:mu] - x)[:, None]
                                 @ (ys_sorted[:mu] - x)[None, :], axis=0))
            # Symmetrize and ensure positive-definiteness.
            C = 0.5 * (C + C.T)
            # Add small regularization to keep it PD.
            C += 1e-9 * np.eye(dim)
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_chol = np.linalg.cholesky(C + 1e-4 * np.eye(dim))

            # Move the mean.
            x = y_mean

        # ---- Final high-precision polish pass (local exact solve) ----
        # Use the best point found and a tiny local (1+1)-ES / coordinate
        # refinement to squeeze out the last bits near the optimum.
        if time.time() < deadline - 1.0:
            px = best_x.copy()
            pfx = best_f
            p_sigma = max(sigma * 0.1, 1e-4)
            for _ in range(200):
                if time.time() > deadline - 1.0:
                    break
                if calls >= n:
                    break
                y = _clip(px + p_sigma * rng.standard_normal(dim))
                fy = func(y)
                calls += 1
                if fy < pfx:
                    px, pfx = y, fy
                    p_sigma *= 1.3
                else:
                    p_sigma *= 0.7
                if p_sigma < 1e-6:
                    break
            if pfx < best_f:
                best_f, best_x = pfx, px

        return best_x, best_f
# EVOLVE-BLOCK-END
