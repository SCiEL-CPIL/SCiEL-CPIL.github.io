"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal Natural Gradient Descent (Rank-1 + Rank-mu) for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 270  # safety margin vs 300s / 330s kill

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(20240611)

        # ---- population / weighting ----
        lambda_ = 14
        mu_ = 7
        w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # ---- strategy constants (adapted for diagonal) ----
        # c_sigma and c_c are similar to CMA-ES
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1.0) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        
        # For diagonal, we use separate learning rates for each dimension
        # c_i is the learning rate for the i-th diagonal element
        # Standard CMA-ES c1 and cmu are split or adapted. 
        # Here we use a common rate for all diagonal elements initially, 
        # but update them independently based on local gradient info.
        c_diag = 2 / (dim ** 2 + 6)  # Similar to c1
        c_mu_diag = min(1 - c_diag, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # ---- state ----
        # x_mean: current mean
        # sigma: global step size (scalar)
        # d: diagonal step sizes (vector of length dim), representing sqrt(C_ii)
        # p_sigma: evolution path for sigma (in standard normal space)
        # p_c: evolution path for mean (in standardized space)
        
        x_mean = np.zeros(dim)
        sigma = 2.0
        d = np.ones(dim)  # Initial diagonal steps are 1 (isotropic)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(y)
            call_count += 1
            if f < best_f:
                best_f = float(f)
                best_x = y.copy()
            return f

        # ---- initial diverse exploration ----
        n_init = min(16, budget)
        for k in range(n_init):
            if timed_out() or call_count >= budget:
                break
            if k == 0:
                x0 = np.zeros(dim)
            elif k == 1:
                x0 = 3.0 * rng.standard_normal(dim)
            elif k == 2:
                x0 = np.linspace(-4.0, 4.0, dim)
            else:
                x0 = 4.0 * rng.uniform(-1.0, 1.0, dim)
            evaluate(x0)

        # ---- main loop: Diagonal Natural Gradient Descent ----
        while call_count < budget:
            if timed_out():
                break

            # generate population
            # y_i = x_mean + sigma * (d * z_i)
            offspring = []
            for _ in range(lambda_):
                if call_count >= budget or timed_out():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (d * z)
                f = evaluate(y)
                offspring.append((f, y, z))

            if len(offspring) < mu_:
                break

            offspring.sort(key=lambda t: t[0])
            selected = offspring[:mu_]

            # update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * selected[i][1] for i in range(mu_)], axis=0)
            delta = x_mean - old_mean

            # Update evolution paths
            # p_c is updated in the standardized space. 
            # In CMA-ES, p_c += c_c * sqrt(mu_eff) * C^{-1/2} (x_new - x_old)
            # Here, C is diagonal with entries (sigma * d_i)^2.
            # So C^{-1/2} delta = delta / (sigma * d)
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (delta / (sigma * d))

            # p_sigma is in z-space
            z_mean = np.sum([w[i] * selected[i][2] for i in range(mu_)], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z_mean

            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-10, 20.0))

            # Update diagonal steps d
            # This is the key structural change.
            # We want d_i to adapt to the local curvature in dimension i.
            # A simple rank-1 + rank-mu update for the diagonal:
            # C_ii_new = (1 - c_diag - c_mu_diag) * C_ii_old + c_diag * (drift_i * p_c_i^2 * C_ii_old) + ...
            # Actually, it's easier to think in terms of step sizes.
            # Let's use a multiplicative update based on the variance of the selected offspring in each dimension.
            
            # Calculate the weighted variance of the selected offspring in each dimension (standardized)
            # var_i = sum(w_k * z_ki^2)
            # If var_i > 1, the distribution was too narrow in dimension i (or the landscape is flat), so we can increase d_i.
            # If var_i < 1, the distribution was too wide, so we should decrease d_i.
            # This is similar to the sigma update but per dimension.
            
            var_z = np.sum(w[:, np.newaxis] * np.array([s[2] for s in selected])[np.newaxis, :] ** 2, axis=0)
            
            # Update d_i
            # We use a learning rate for the diagonal updates.
            # Let's use c_diag for this.
            # d_i *= exp(c_diag * (var_z_i - 1))
            # This is a natural gradient step for the log-diagonal.
            
            # However, we also need to account for the drift term (p_c).
            # The drift term in CMA-ES corrects for the bias in the covariance update.
            # For diagonal, the drift is p_c_i^2.
            # The expected value of var_z is 1.
            # The update should be:
            # log(d_i) += c_diag * (var_z_i - 1) + ... drift correction ...
            
            # Let's stick to a simpler, robust update:
            # d_i = d_i * exp(c_diag * (var_z_i - 1))
            # This is a multiplicative update that adapts the step size in each dimension independently.
            
            d = d * np.exp(c_diag * (var_z - 1))
            d = np.clip(d, 1e-4, 100.0)

            # Stagnation guard: if sigma is very small and d is very small, re-seed
            if sigma < 1e-7 and np.mean(d) < 1e-3:
                x_mean = best_x + 0.3 * rng.standard_normal(dim)
                sigma = 0.5
                d = np.ones(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)

        # ---- final polish ----
        if call_count < budget and not timed_out() and best_x is not None:
            remaining = budget - call_count
            n_polish = min(remaining, 50)
            if n_polish > 0:
                scale = max(sigma * np.mean(d) * 0.5, 1e-4)
                for i in range(n_polish):
                    if timed_out() or call_count >= budget:
                        break
                    direction = rng.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-15)
                    for step_scale in [scale, scale * 0.1]:
                        if call_count >= budget or timed_out():
                            break
                        candidate = best_x + step_scale * direction
                        candidate = np.clip(candidate, -5.0, 5.0)
                        f = evaluate(candidate)
                        if f >= best_f:
                            candidate2 = best_x - step_scale * direction
                            candidate2 = np.clip(candidate2, -5.0, 5.0)
                            f2 = evaluate(candidate2)
                            if f2 < best_f:
                                best_f = f2
                                best_x = candidate2.copy()
                            break
                        else:
                            best_f = f
                            best_x = candidate.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
