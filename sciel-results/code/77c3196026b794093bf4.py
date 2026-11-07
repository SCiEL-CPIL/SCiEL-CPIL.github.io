"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with coordinate-descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 30 s safety margin before 320 s hard kill

        def elapsed():
            return time.time() - start_time

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Multi-start CMA-ES ---
        # 5 starts: center + 4 random points in [-4, 4]^dim
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))

        # Budget allocation: center gets more evals
        n_starts = len(starts)
        base_budget = 60
        center_budget = 200
        total_cma = center_budget + (n_starts - 1) * base_budget

        # If total_cma exceeds budget, scale down
        if total_cma >= budget:
            scale = (budget - 50) / total_cma  # leave 50 for polish
            center_budget = max(10, int(center_budget * scale))
            base_budget = max(10, int(base_budget * scale))
            total_cma = center_budget + (n_starts - 1) * base_budget

        # Polish budget
        polish_budget = max(20, budget - total_cma - 5)

        best_x = None
        best_f = np.inf

        def cma_es(x0, n_evals):
            """Run CMA-ES from x0 with n_evals function evaluations."""
            nonlocal best_x, best_f
            d = dim
            pop_size = max(4, int(4 + 3 * np.log(d)))
            sigma = 1.0
            C = np.eye(d)
            # CMA-ES weights
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            ps = np.zeros(d)
            x_mean = x0.copy().astype(float)
            # Clip to box
            x_mean = np.clip(x_mean, -5, 5)

            f_values = []
            x_samples = []
            calls = 0

            for _ in range(n_evals // pop_size + 1):
                if elapsed() > deadline:
                    break
                # Generate population
                for i in range(pop_size):
                    if elapsed() > deadline:
                        break
                    z = rng.standard_normal(d)
                    x_trial = np.clip(x_mean + sigma * (C @ z), -5, 5)
                    f_trial = func(x_trial)
                    calls += 1
                    x_samples.append(x_trial.copy())
                    f_values.append(f_trial)
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()

                if len(x_samples) < pop_size:
                    break

                # Sort by fitness
                order = np.argsort(f_values)
                x_sorted = [x_samples[i] for i in order]
                f_sorted = [f_values[i] for i in order]

                # Update mean
                x_old = x_mean.copy()
                x_mean = sum(w * x for w, x in zip(weights, x_sorted[:mu]))
                x_mean = np.clip(x_mean, -5, 5)

                # Update evolution path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma) * (C @ (x_mean - x_old))
                hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // pop_size + 1))) / chi_d
                if hs < 1.4 + 2 / (d + 1):
                    C = (1 - c_c) * C + c_c * (2 + c_c) * np.outer(ps, ps)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1))
                sigma = np.clip(sigma, 1e-12, 50.0)

                # Update covariance
                delta_hs = (1.0 if hs < 1.4 + 2 / (d + 1) else 0.0) - 1.0
                rank_mu = np.zeros((d, d))
                for i in range(mu):
                    y_i = (x_sorted[i] - x_old) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                C = (1 - c_c * (2 + c_c - c_c) - c_c * delta_hs - c_c * mu_eff) * C \
                    + c_c * (2 + c_c - c_c + c_c * delta_hs) * np.outer(ps, ps) \
                    + c_c * mu_eff * rank_mu

                # Symmetrize
                C = 0.5 * (C + C.T)

                x_samples = []
                f_values = []

            return calls

        # Run multi-start CMA-ES
        for i, x0 in enumerate(starts):
            if elapsed() > deadline:
                break
            n_evals = center_budget if i == 0 else base_budget
            cma_es(x0, n_evals)

        # --- Coordinate descent polish ---
        if best_x is not None and elapsed() < deadline:
            x = best_x.copy()
            f = best_f
            step = 0.5
            for _ in range(polish_budget):
                if elapsed() > deadline:
                    break
                improved = False
                for d_idx in range(dim):
                    if elapsed() > deadline:
                        break
                    for direction in (-1, 1):
                        x_trial = x.copy()
                        x_trial[d_idx] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            best_x = x.copy()
                            best_f = f
                            improved = True
                            break  # only one step per coordinate per iteration
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break
            return x, f

        # Fallback: return best found
        if best_x is not None:
            return best_x, best_f
        # Should not reach here, but safety
        x = np.zeros(dim)
        f = func(x)
        return x, f

# EVOLVE-BLOCK-END
