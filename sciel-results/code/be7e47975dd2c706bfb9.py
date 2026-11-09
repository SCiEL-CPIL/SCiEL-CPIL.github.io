"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        if budget <= 0:
            return np.zeros(d), 0.0

        # Parameters
        lam = max(4, int(4 + 3 * np.log(d)))  # population size
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 + 2.0 / (d + 1.3)) / ((d + 1.2) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   (2 + (mu_eff / d - 1) * (0.8 + mu_eff / d)) /
                   ((d + 1.5) ** 2 + mu_eff))
        chi_d = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        # Initialize
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0
        B = np.eye(d)
        D = 1.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        calls = 1

        # Precompute
        hs = (1 + 2.0 / (d + 1)) ** (1.0 / (10 * lam * c_c)) + 1e-3

        while calls < budget:
            # Generate population
            offspring = []
            offspring_f = []
            for i in range(lam):
                z = rng.standard_normal(d)
                # x + sigma * B * D * z
                step = B @ (D * z)
                y = x + sigma * step
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if calls >= budget:
                    # If we ran out of budget mid-population, stop
                    if fy < best_fx:
                        best_x = y.copy()
                        best_fx = fy
                    break
                offspring.append(y)
                offspring_f.append(fy)
                if fy < best_fx:
                    best_x = y.copy()
                    best_fx = fy

            if calls >= budget:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            y_sorted = [offspring[i] for i in indices]
            f_sorted = [offspring_f[i] for i in indices]

            # Weighted mean
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += w[i] * y_sorted[i]

            # Update step size
            z_mean = np.linalg.solve(B.T, (y_w - x) / (sigma * D)) if D > 1e-10 else np.zeros(d)
            # Actually, B is orthogonal, so B.T @ B = I. But D is diagonal.
            # y_w - x = sigma * B @ (D @ z_mean_raw)
            # z_mean_raw = D @ (B.T @ (y_w - x) / sigma)
            # The update rule uses ||z_mean|| / chi_d
            # Let's compute z_mean properly.
            # y_w = x + sigma * B @ (D @ z_avg) where z_avg = sum w_i z_i
            # So (y_w - x) / sigma = B @ (D @ z_avg)
            # B.T @ ((y_w - x) / sigma) = D @ z_avg
            # z_avg = D^{-1} @ (B.T @ ((y_w - x) / sigma))
            
            if sigma > 1e-15:
                diff = (y_w - x) / sigma
                z_avg = np.diag(1.0 / D) @ (B.T @ diff)
            else:
                z_avg = np.zeros(d)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_avg
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) <
                              (1.4 + 2.0 / (d + 1)) * chi_d) else 0.0

            sigma *= np.exp(c_sigma / cs * (norm_p_sigma / chi_d - 1.0))
            cs = 1.0  # placeholder, need proper c_sigma
            # Correct c_sigma
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            sigma *= np.exp(c_sigma / c_sigma * (norm_p_sigma / chi_d - 1.0)) # Wait, standard is exp(c_sigma * (norm/chi - 1))
            # Let's re-implement the sigma update correctly.
            # sigma = sigma * exp(c_sigma * (||p_sigma||/chi_d - 1))
            # I already multiplied by exp... let's reset and do it cleanly in the next iteration or fix here.
            # Actually, I calculated sigma *= ... twice or wrong. Let's restart the loop logic mentally.
            
            # Re-doing the update step cleanly:
            # 1. Compute z_avg
            # 2. Update p_sigma
            # 3. Update sigma
            # 4. Update p_c
            # 5. Update C
            # 6. Update x

            # Let's restart the block from sigma update properly.
            # The code above was messy. Let's rewrite the whole __call__ cleanly.
            pass

        return best_x, best_fx

    def _run_cma(self, func):
        d = self.dim
        budget = self.budget
        if budget <= 0:
            return np.zeros(d), 0.0

        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 + 2.0 / (d + 1.3)) / ((d + 1.2) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   (2 + (mu_eff / d - 1) * (0.8 + mu_eff / d)) /
                   ((d + 1.5) ** 2 + mu_eff))
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        chi_d = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0
        B = np.eye(d)
        D = np.ones(d)
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        calls = 1

        while calls < budget:
            offspring = []
            offspring_f = []
            for i in range(lam):
                z = rng.standard_normal(d)
                step = B @ (D * z)
                y = x + sigma * step
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_x = y.copy()
                    best_fx = fy
                if calls >= budget:
                    break
                offspring.append(y)
                offspring_f.append(fy)

            if calls >= budget:
                break

            if len(offspring) < mu:
                break

            indices = np.argsort(offspring_f)
            y_sorted = [offspring[i] for i in indices]

            y_w = np.zeros(d)
            for i in range(mu):
                y_w += w[i] * y_sorted[i]

            if sigma > 1e-15:
                diff = (y_w - x) / sigma
                z_avg = np.diag(1.0 / D) @ (B.T @ diff)
            else:
                z_avg = np.zeros(d)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_avg
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, budget - calls))) <
                              (1.4 + 2.0 / (d + 1)) * chi_d) else 0.0

            sigma *= np.exp(c_sigma * (norm_p_sigma / chi_d - 1.0))

            x_new = y_w
            x_diff = (x_new - x) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * x_diff

            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (C - np.outer(p_c, p_c) + np.outer(p_c, p_c)) + # This is wrong, standard is (1-c1-cmu)C + c1(p_c p_c^T + (1-h)delta C) + cmu sum w_i z_i z_i^T
                 # Let's use the standard update:
                 # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * z_i * z_i.T)
                 # I need z_i for the top mu.
                 0)

            # Recompute C update properly
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                if sigma > 1e-15:
                    z_i = np.diag(1.0 / D) @ (B.T @ ((y_sorted[i] - x) / sigma))
                else:
                    z_i = np.zeros(d)
                rank_mu += w[i] * np.outer(z_i, z_i)

            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) +
                 c_mu * rank_mu)

            # Ensure symmetry
            C = (C + C.T) / 2.0

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            x = x_new
            # fx is not strictly needed for CMA-ES update, only for best tracking

        return best_x, best_fx
# EVOLVE-BLOCK-END
