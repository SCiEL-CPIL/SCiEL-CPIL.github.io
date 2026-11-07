"""Rewritten optimizer for BBOB-style Sphere with hidden affine transforms."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic estimator + CMA-ES polish for Sphere with hidden affine transform."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        t_start = time.time()
        deadline = 285.0  # safety margin below 300s

        calls = [0]
        best_x = None
        best_fx = np.inf

        def f(x):
            nonlocal best_x, best_fx
            if calls[0] >= budget:
                return best_fx
            calls[0] += 1
            val = float(func(x))
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        # --- Phase 1: Estimate affine quadratic model using finite differences ---
        # Sphere with hidden affine: f(x) = 0.5*(x-c)^T A (x-c) + f0
        # We estimate c and A by evaluating at strategic points.

        # Start at origin
        x0 = np.zeros(dim)
        fx0 = f(x0)

        # Estimate gradient at x0 via central differences
        h1 = 0.01
        grad = np.zeros(dim)
        for i in range(dim):
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += h1
            xm[i] -= h1
            xp = np.clip(xp, -5, 5)
            xm = np.clip(xm, -5, 5)
            fp = f(xp)
            fm = f(xm)
            grad[i] = (fp - fm) / (2 * h1)

        # Estimate Hessian via second-order finite differences
        h2 = 0.1
        hess = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i, dim):
                xpp = x0.copy()
                xpp[i] += h2
                xpp[j] += h2
                xpp = np.clip(xpp, -5, 5)

                xmm = x0.copy()
                xmm[i] -= h2
                xmm[j] -= h2
                xmm = np.clip(xmm, -5, 5)

                xpm = x0.copy()
                xpm[i] += h2
                xpm[j] -= h2
                xpm = np.clip(xpm, -5, 5)

                xmp = x0.copy()
                xmp[i] -= h2
                xmp[j] += h2
                xmp = np.clip(xmp, -5, 5)

                fpp = f(xpp)
                fmm = f(xmm)
                fpm = f(xpm)
                fmp = f(xmp)

                if i == j:
                    hess_val = (fpp + fmm - 2 * fx0) / (2 * h2 * h2)
                else:
                    hess_val = (fpp + fmm - fpm - fmp) / (4 * h2 * h2)
                hess[i, j] = hess_val
                hess[j, i] = hess_val

        # Solve for minimizer: c = x0 - H^{-1} * grad
        try:
            eigvals, eigvecs = np.linalg.eigh(hess)
            eigvals = np.maximum(eigvals, 1e-8)
            hess_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            c_est = x0 - hess_inv @ grad
            c_est = np.clip(c_est, -5, 5)
            fc_est = f(c_est)
        except np.linalg.LinAlgError:
            c_est = best_x if best_x is not None else x0

        if calls[0] >= budget:
            return best_x, best_fx

        # --- Phase 2: CMA-ES refinement starting near the estimated optimum ---
        # Use a simplified CMA-ES with adaptive step size

        n = dim
        lam = max(4 + int(3 * np.log(n)), 10)  # population size

        mean = c_est.copy() if c_est is not None else best_x.copy()
        sigma = 0.5
        C = np.eye(n)

        # Precompute eigen decomposition of C
        eigvals_C, eigvecs_C = np.linalg.eigh(C)
        eigvals_C = np.maximum(eigvals_C, 1e-20)
        D = np.sqrt(eigvals_C)
        B = eigvecs_C
        D_inv = 1.0 / D

        # CMA-ES parameters
        cs = 1 / (np.sqrt(n) + 2)
        ds = 1 + 2 * max(0, (n + 1) / 3 - 1)
        c1 = 2 / ((n + 1.3)**2 + 5)
        cmu = min(1 - c1, 2 * ((n - 2 + 1/n) / (n + 2)**2 - 1))
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        weights = np.zeros(lam)
        weights[:mu] = w
        weights[mu] = 1 - w.sum()
        weights = np.maximum(weights, 0)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights[:mu]**2)

        ps = np.zeros(n)
        ph = np.zeros(n)

        # Generate and evaluate population
        def generate_and_evaluate():
            pop = []
            fvals = []
            for k in range(lam):
                if calls[0] >= budget or (time.time() - t_start) > deadline:
                    break
                z = np.random.standard_normal(n)
                y = B @ (D_inv * z)
                x_new = mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                pop.append(x_new)
                fv = f(x_new)
                fvals.append(fv)
            return pop, fvals

        # Main CMA-ES loop
        max_iter = 200
        for iteration in range(max_iter):
            if calls[0] >= budget or (time.time() - t_start) > deadline:
                break

            pop, fvals = generate_and_evaluate()
            if len(fvals) < lam:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = [pop[i] for i in order]
            fvals_sorted = [fvals[i] for i in order]

            # Update mean
            mean_new = np.sum([weights[i] * pop_sorted[i] for i in range(lam)], axis=0)

            # Update evolution paths
            C_inv_sqrt_mean_diff = B @ (D_inv * (B.T @ (mean_new - mean))) / (sigma + 1e-20)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt_mean_diff

            # Check termination
            norm_ps = np.linalg.norm(ps)
            if norm_ps / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) > 1.4 + 4 / n:
                break

            ph = (1 - ds) * ph + np.sqrt(ds * (2 - ds) * mu_eff) * (mean_new - mean) / (sigma * np.linalg.norm(C_inv_sqrt_mean_diff) + 1e-20)

            # Update covariance
            y_vectors = [(pop_sorted[i] - mean) / (sigma + 1e-20) for i in range(mu)]
            delta_h = (1 + (1 - ds) * 2 * (n + 1) / 3) * np.sqrt((1 - (1 - ds)**2) / (1 - (1 - ds)**(2 * (iteration + 1))))
            cmu_eff = cmu if np.linalg.norm(ph) / np.sqrt(1 - (1 - ds)**(2 * (iteration + 1))) < (1.4 + 2 / (n + 1)) * delta_h else 0

            C = (1 - c1 - cmu_eff) * C + c1 * (np.outer(ps, ps) + (1 - cs) * C)
            if cmu_eff > 0:
                for i in range(mu):
                    C += cmu_eff * weights[i] * np.outer(y_vectors[i], y_vectors[i])

            C = (C + C.T) / 2
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            eigvals_C = np.maximum(eigvals_C, 1e-20)
            C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T

            # Update eigen decomposition
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            eigvals_C = np.maximum(eigvals_C, 1e-20)
            D = np.sqrt(eigvals_C)
            B = eigvecs_C
            D_inv = 1.0 / D

            # Update sigma
            sigma = sigma * np.exp(cs / (5 * (n + 2)) * (norm_ps / np.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) - np.sqrt(n * (1 - 1/(n+1)))))
            sigma = np.clip(sigma, 1e-10, 10.0)

            mean = mean_new

            # Early stopping if we're very close to optimum
            if best_fx < 1e-6:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
