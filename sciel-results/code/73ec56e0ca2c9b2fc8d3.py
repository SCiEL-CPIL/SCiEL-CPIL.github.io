"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES
        x = np.zeros(dim)
        f = func(x)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + dim / 3) / (dim + 4 + 2 / (dim + 1))
        cs = (dim + 1.7 - mu_w) / (dim + cs0 + 2) if (dim + 1.7 - mu_w) > 0 else 0.8
        # Standard CMA-ES parameters
        mu_w = dim // 2
        w = np.log(np.arange(1, mu_w + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c_sigma = (5 + mu_eff) / (dim + 5 + 3 * mu_eff)
        c1 = (2 - mu_eff / (dim + 1.3) ** 2 + 2 * cc) / ((dim + 2) * (1 - cc))
        cmu = (2 * (mu_w - 1 + w[mu_w - 1]) / (dim + 1.3) ** 2 - c1) / ((dim + 2) * (1 - cc))
        cmu = max(0, min(cmu, 2 * (1 - c1) / (dim + 1.3) ** 2))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        damps = (1 + 2 * max(0, np.sqrt((mu_w - 1) / (dim + 1)) - 1) + cs)

        calls = 1
        best_x = x.copy()
        best_f = f

        while calls < budget:
            # Time check
            if time.time() > deadline:
                break

            # Generate lambda candidates
            lam = max(4, dim + 1)
            y = np.random.randn(lam, dim)
            # Update covariance approximation for step size
            y = y / np.linalg.norm(y, axis=1, keepdims=True) * sigma * np.sqrt(np.diag(C).mean())
            # Simplified: just use sigma * random normal
            y = np.random.randn(lam, dim) * sigma

            # Clip to bounds
            x_candidates = np.clip(x[None, :] + y, -5, 5)

            # Evaluate
            new_points = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                xi = x_candidates[i]
                fi = func(xi)
                calls += 1
                new_points.append((fi, xi))

            if not new_points:
                break

            # Sort by fitness
            new_points.sort(key=lambda t: t[0])
            top_k = min(mu_w, len(new_points))
            if top_k == 0:
                break

            # Update best
            if new_points[0][0] < best_f:
                best_f = new_points[0][0]
                best_x = new_points[0][1].copy()

            # Compute weighted mean of top mu
            y_mean = np.zeros(dim)
            for i in range(top_k):
                fi, xi = new_points[i]
                y_mean += w[i] * (xi - x)
            x_mean = x + y_mean

            # Update path variables
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * np.dot(C, y_mean)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (x_mean - x) / sigma

            # Update covariance
            delta_hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) / (dim + cs))) > (1.4 + 2 / (dim + 1)) * chi_n)
            C = (1 + c1 + cmu - c1 - cs * (1 - cc) + (delta_hs - 1) * c1 * cc * (1 - cc)) * C
            C += c1 * np.outer(pc, pc)
            for i in range(top_k):
                zi = (new_points[i][1] - x) / sigma
                C += cmu * w[i] * np.outer(zi, zi)

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            x = x_mean

            # Ensure x is within bounds
            x = np.clip(x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
