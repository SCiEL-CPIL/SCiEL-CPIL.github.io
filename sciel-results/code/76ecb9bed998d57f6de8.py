"""CMA-ES optimizer for BBOB f12 (Bent Cigar), 5D, box [-5,5]^5."""
import time
import numpy as np


class Optimizer:
    """CMA-ES with a small multi-start for the shifted/rotated Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0

        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff + 13)
        cmu = (
            2 * ((mu_eff + 2) / (dim + 2) - c1)
            / ((dim + 1) ** 2 + mu_eff + 16)
        )
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        def cma_es(mean, sigma, remaining_budget, rng):
            if remaining_budget < lam + 1:
                f0 = func(mean)
                return mean, f0

            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            best_x = mean.copy()
            best_f = func(mean)
            count = 1

            while count + lam <= remaining_budget and time.time() < deadline:
                # Eigendecomposition of C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_eig = np.sqrt(eigvals)
                inv_sqrt_eig = 1.0 / sqrt_eig

                # Generate population
                samples = []
                for _ in range(lam):
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (eigvecs @ (sqrt_eig * z))
                    y = np.clip(y, -5.0, 5.0)
                    samples.append(y)

                fvals = np.array([func(y) for y in samples])
                count += lam

                idx = np.argsort(fvals)
                samples_sorted = [samples[i] for i in idx]
                fvals_sorted = fvals[idx]

                if fvals_sorted[0] < best_f:
                    best_f = fvals_sorted[0]
                    best_x = samples_sorted[0].copy()

                # New mean
                new_mean = np.sum(w[:, None] * np.array(samples_sorted[:mu]), axis=0)

                # Step size control
                y_sigma = (eigvecs @ (inv_sqrt_eig * ((new_mean - mean) / sigma)))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                    c_sigma * (2 - c_sigma) * mu_eff
                ) * y_sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sigma = norm_p_sigma / chi_n < 1.4 + 2.0 / (dim + 1)
                if h_sigma:
                    sigma *= np.exp(
                        c_sigma / d_sigma * (norm_p_sigma / chi_n - 1)
                    )
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 20.0)

                # Covariance update
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(
                    c_c * (2 - c_c) * mu_eff
                ) * (new_mean - mean) / sigma
                rank1 = np.outer(p_c, p_c)
                delta_h = (1 - c_c) ** (2 * (remaining_budget - count))
                rankmu = np.sum(
                    w[:, None, None] * np.outer(
                        (np.array(samples_sorted[:mu]) - mean) / sigma,
                        (np.array(samples_sorted[:mu]) - mean) / sigma,
                    ),
                    axis=0,
                )
                C = (
                    (1 - c1 - cmu + c1 * (1 - delta_h)) * C
                    + c1 * rank1
                    + cmu * rankmu
                )
                C = 0.5 * (C + C.T)
                C += 1e-12 * np.eye(dim)

                mean = new_mean

            return best_x, best_f

        rng = np.random.default_rng(42)

        # Multi-start: center + one random point
        starts = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
        ]

        budget1 = int(0.7 * budget)
        budget2 = budget - budget1

        best_x = None
        best_f = np.inf

        for i, start in enumerate(starts):
            rem = budget1 if i == 0 else budget2
            if rem < lam + 1:
                continue
            bx, bf = cma_es(start, 2.0, rem, rng)
            if bf < best_f:
                best_f = bf
                best_x = bx
            if time.time() > deadline - 30:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
