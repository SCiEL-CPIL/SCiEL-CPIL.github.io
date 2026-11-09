"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 330s kill - 50s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Split budget among restarts
        n_restarts = 5
        calls_per_restart = budget // n_restarts  # 100
        lam = 8  # population size per generation

        global_best_x = None
        global_best_f = np.inf

        calls_used = 0

        for restart in range(n_restarts):
            # Check time
            if time.time() >= deadline:
                break
            if calls_used >= budget:
                break

            # Initialize: for first restart use center, others random
            if restart == 0:
                x_mean = np.zeros(dim)
            else:
                x_mean = rng.uniform(-4, 4, dim)

            # Initial sigma: large for exploration
            sigma = 3.0

            # CMA-ES state
            C = np.eye(dim)
            # Step size control
            cs = 1 + 2 * min((lam // 2 - 1) / dim, 1) / dim
            ds = cs + 2 * min((lam // 2 - 1) / dim, 1) + 1 / dim
            c_sigma = (cs + 1) / (dim + 7.5)
            c_c = (4 + dim / lam) / (dim + 4 + 2 * lam / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + lam)
            c_mu = min(1 - c_1, 2 * ((mu_o + 1) / (dim * lam) - 1))
            mu_o = lam // 2
            weights = np.log(mu_o + 1) - np.log(np.arange(1, mu_o + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Generate population for this restart
            gen_calls = 0
            max_gen_calls = calls_per_restart - (restart * calls_per_restart)  # remaining
            # Actually just use calls_per_restart for each restart
            max_gen_calls = calls_per_restart

            while gen_calls < max_gen_calls:
                if time.time() >= deadline:
                    break
                if calls_used >= budget:
                    break

                # Check if we can fit a full generation
                remaining = max_gen_calls - gen_calls
                if remaining < lam:
                    # Do a final partial generation or just break
                    # Better to break and let next restart handle it
                    break

                # Sample population
                pop = np.zeros((lam, dim))
                fvals = np.zeros(lam)

                for i in range(lam):
                    if time.time() >= deadline or calls_used >= budget:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(x_mean + sigma * z, -5, 5)
                    fy = func(y)
                    pop[i] = y
                    fvals[i] = fy
                    calls_used += 1
                    gen_calls += 1

                    if fy < global_best_f:
                        global_best_f = fy
                        global_best_x = y.copy()

                if gen_calls >= max_gen_calls or calls_used >= budget:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                fvals_sorted = fvals[order]
                pop_sorted = pop[order]

                # Update best
                if fvals_sorted[0] < global_best_f:
                    global_best_f = fvals_sorted[0]
                    global_best_x = pop_sorted[0].copy()

                # Weighted mean
                x_new = np.zeros(dim)
                for i in range(mu_o):
                    x_new += weights[i] * pop_sorted[i]

                # Update covariance
                diff = x_new - x_mean
                # Normalize
                if np.linalg.norm(diff) > 1e-12:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / (sigma * np.linalg.norm(diff))
                x_mean = x_new

                # Update step size
                norm_p_sigma = np.linalg.norm(p_sigma)
                chi_lam = np.sqrt(lam) * (1 - 1 / (4 * lam) - 1 / (24 * lam ** 2))
                sigma *= np.exp(c_sigma / ds * (norm_p_sigma / chi_lam - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                # Update C
                # Compute rank-one and rank-mu updates
                diff_scaled = diff / sigma
                C = (1 - c_c * (c_1 + c_mu)) * C \
                    + c_c * c_1 * np.outer(diff_scaled, diff_scaled) \
                    + c_c * c_mu * np.sum(weights[:mu_o] * np.outer(pop_sorted[:mu_o] - x_mean, pop_sorted[:mu_o] - x_mean), axis=0) / (sigma ** 2)

                # Ensure C is symmetric PSD (numerical safety)
                C = (C + C.T) / 2
                # Eigenvalue clipping for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
