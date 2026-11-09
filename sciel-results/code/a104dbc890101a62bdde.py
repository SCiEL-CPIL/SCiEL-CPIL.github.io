"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        popsize = 4
        mu = popsize // 2  # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize
        x_mean = np.zeros(dim)
        sigma = 5.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # not needed directly but for clarity
        # We need sqrt of C for sampling. Use Cholesky decomposition.
        # Initially C = I, so L = I
        L = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls_used = 1

        while calls_used < budget:
            # Generate population
            xs = np.zeros((popsize, dim))
            for i in range(popsize):
                z = rng.standard_normal(dim)
                # Transform z: x = x_mean + sigma * L @ z
                xs[i] = x_mean + sigma * (L @ z)
                # Clip to box
                xs[i] = np.clip(xs[i], -5, 5)

            # Evaluate
            fs = np.zeros(popsize)
            for i in range(popsize):
                fs[i] = func(xs[i])
                calls_used += 1

            # Track best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = xs[idx_min].copy()

            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]

            # Update mean
            x_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)

            # Update path for covariance (pc)
            # Need C^{-1/2} (x_new - x_mean) / sigma
            # Compute C^{-1/2} via Cholesky: C = L L^T, so C^{-1/2} = (L^{-1})^T
            # Actually, C^{-1/2} applied to v is equivalent to solving L^T L w = v => w = L^{-T} v
            # Let's compute inv_L = np.linalg.inv(L)
            try:
                inv_L = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback: reset C to identity
                C = np.eye(dim)
                L = np.eye(dim)
                inv_L = np.eye(dim)

            # C^{-1/2} v = inv_L^T v
            diff = (x_new - x_mean) / sigma
            pc = (1 - c_c) * pc + c_c * (inv_L.T @ diff)

            # Update path for sigma (ps)
            # Need C^{-1/2} (x_new - x_mean) / sigma for the norm
            # hsig = 1 if norm(pc) / sqrt(1 - (1-c_c)^{2*(budget-calls_used+1)}) < (1.4 + 2/(dim+1)) * chi_n
            # For simplicity, use a fixed threshold or approximate
            norm_pc = np.linalg.norm(pc)
            # Approximate the threshold
            # theta = (1.4 + 2 / (dim + 1)) * chi_n
            # We don't track generations exactly, so use a simple heuristic
            # Actually, let's compute it properly
            # generations = budget - calls_used + popsize  # approximate
            # Let's just use a fixed hsig based on typical behavior
            hsig = 1.0  # Simplified; could be improved
            ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma)) * (inv_L.T @ diff)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - hsig) * c_c * (2 - c_c) * I) + c_mu * sum(weights_i * (x_i - x_mean)(x_i - x_mean)^T)
            # The term (x_i - x_mean) should be transformed by C^{-1/2}
            # Let y_i = C^{-1/2} (x_i - x_mean) / sigma
            # Then C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + ...) + c_mu sum(w_i y_i y_i^T) * sigma^2 ?
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - hsig) c_c (2 - c_c) I) + c_mu sum_i w_i (y_i y_i^T)
            # where y_i = C^{-1/2} (x_i - x_mean) / sigma
            # But wait, the update is in the transformed space. Let me re-derive.
            # Standard CMA-ES:
            # C ← (1 − c1 − cmu) C + c1 (pc pc^T + (1 − hsig) cc (2 − cc) I) + cmu Σ wi yi yi^T
            # where yi = C^{-1/2} (xi − xmean) / σ
            # This is correct.

            # Compute y_i for the mu best individuals
            y_terms = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (xs_sorted[i] - x_mean) / sigma
                y_i = inv_L.T @ diff_i
                y_terms += weights[i] * np.outer(y_i, y_i)

            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * np.eye(dim)) + c_mu * y_terms

            # Symmetrize C
            C = (C + C.T) / 2

            # Compute new L (Cholesky of C)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    C = np.eye(dim)

            # Update mean
            x_mean = x_new

            # Clip x_mean to box (optional, but helps)
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
