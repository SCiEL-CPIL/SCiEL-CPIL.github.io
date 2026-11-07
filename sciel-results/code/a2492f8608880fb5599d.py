"""Optimizer for BBOB f14 (Different powers) with CMA-ES and coordinate refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with adaptive strategy for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.full(dim, 5.0)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 2.0

        def calls_left():
            return budget - calls

        rng = np.random.default_rng(12345)

        # --- Phase 1: CMA-ES with population ---
        # Initial population centered at origin with moderate spread
        pop_size = 12
        x_mean = np.zeros(dim)
        sigma = 2.0

        # Generate initial population
        pop = []
        for i in range(pop_size):
            x_i = np.clip(x_mean + sigma * rng.standard_normal(dim), -5, 5)
            f_i = func(x_i)
            calls += 1
            update_best(x_i, f_i)
            pop.append((f_i, x_i))

        # CMA-ES main loop
        c = 1.0 / (dim + 3)
        d = 1.0 + 2 * max(0, (np.sqrt((dim + 1) / (dim - 1)) - 1)) / (pop_size + 3)
        c_sigma = (pop_size + 2) / (dim + 2) ** 1.7
        c_sigma = 1.0 / (pop_size + 3) ** 1.7
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        step_size = np.sqrt((1 - c_c) * (2 + c_c) * pop_size / 4)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C_inv = np.eye(dim)
        C = np.eye(dim)

        # Sort initial population
        pop.sort(key=lambda t: t[0])

        for gen in range(200):
            if not time_left() or calls_left() <= pop_size:
                break

            # Select best half
            mu = pop_size // 2
            weights = np.ones(mu)
            weights /= weights.sum()
            mu_eff = 1.0 / (weights @ weights)

            # Update mean
            x_new = x_mean.copy()
            for w, (_, x_i) in zip(weights, pop[:mu]):
                x_new += w * x_i
            x_mean = x_new

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ (x_mean - x_prev_mean)) / sigma
            # Simplified: just adapt sigma based on success rate
            success = sum(1 for _, x_i in pop[:mu] if True)  # placeholder
            # Use simpler adaptation
            if sum(1 for i in range(mu) if pop[i][0] < pop[mu][0]) > mu / 2:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = np.clip(sigma, 0.01, 10.0)

            # Generate new population
            new_pop = []
            for i in range(pop_size):
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Approximate using current C (identity for simplicity, could improve)
                x_i = np.clip(x_mean + sigma * z, -5, 5)
                f_i = func(x_i)
                calls += 1
                update_best(x_i, f_i)
                new_pop.append((f_i, x_i))

            pop.sort(key=lambda t: t[0])
            x_prev_mean = x_mean.copy()

        # --- Phase 2: Coordinate-wise refinement ---
        if time_left() and calls_left() > 10:
            x_curr = best_x.copy()
            f_curr = best_f
            step_sizes = [0.5, 0.1, 0.01, 0.001]

            for step in step_sizes:
                if not time_left() or calls_left() <= 2:
                    break
                improved = True
                while improved and time_left() and calls_left() > 2:
                    improved = False
                    for coord in range(dim):
                        if not time_left() or calls_left() <= 2:
                            break
                        for direction in [1, -1]:
                            if not time_left() or calls_left() <= 2:
                                break
                            trial_x = x_curr.copy()
                            trial_x[coord] = np.clip(x_curr[coord] + direction * step, -5, 5)
                            trial_f = func(trial_x)
                            calls += 1
                            if trial_f < f_curr:
                                x_curr = trial_x
                                f_curr = trial_f
                                update_best(x_curr, f_curr)
                                improved = True
                                break

        # --- Phase 3: Nelder-Mead refinement ---
        if time_left() and calls_left() > 5:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    func, best_x, method='Nelder-Mead',
                    options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-5,
                             'adaptive': True}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
