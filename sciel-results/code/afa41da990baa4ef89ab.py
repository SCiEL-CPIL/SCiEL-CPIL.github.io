"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with time guard for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        # Track best solution
        best_x = np.full(dim, 5.0)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 2.0

        # --- Phase 1: Quick random sampling to get a good initial point ---
        rng = np.random.default_rng(12345)

        # Sample a few random points to find a decent starting point
        n_samples = min(20, budget)
        for _ in range(n_samples):
            if not time_left() or calls >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls += 1
            update_best(x, fx)

        if calls >= budget or not time_left():
            return best_x, best_f

        # --- Phase 2: CMA-ES from the best point found ---
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        # Standard CMA-ES weights
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_1 = (2 / ((dim + 1.3)**2 + mu_eff))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Initial sigma based on best function value spread
        # Use a heuristic: if best_f is large, start with larger sigma
        sigma = min(2.0, max(0.5, 1.0 / (1 + abs(best_f) / 100)))

        while calls < budget and time_left():
            # Generate population
            for i in range(pop_size):
                if not time_left() or calls >= budget:
                    break

                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Apply C matrix (use Cholesky or just diagonal approximation for speed)
                # For simplicity and speed, use a diagonal approximation
                # Actually, let's use a proper approach but keep it fast
                # For dim=5, we can afford the full matrix
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    y = z

                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls += 1
                update_best(x_new, f_new)

                if i == 0:
                    # Store first sample for path updates
                    pass

            if not time_left() or calls >= budget:
                break

            # After generating population, update CMA-ES parameters
            # This is a simplified update - in a full implementation we'd sort by fitness
            # For now, let's do a simpler adaptive step size approach

            # Adaptive sigma based on recent improvement
            if time.time() - t_start > 5:  # Check periodically
                # Reduce sigma if we're making slow progress
                sigma *= 0.95
                sigma = max(sigma, 0.01)

            # Periodically re-center around best found
            if calls % (pop_size * 10) == 0:
                x_mean = best_x.copy()
                sigma = max(0.1, sigma * 0.8)

        # --- Phase 3: Local refinement around best found ---
        if time_left() and calls < budget:
            x_local = best_x.copy()
            f_local = best_f
            sigma_local = 0.1
            max_local_steps = min(100, budget - calls)

            for _ in range(max_local_steps):
                if not time_left() or calls >= budget:
                    break
                y = np.clip(x_local + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < f_local:
                    x_local = y
                    f_local = fy
                    update_best(y, fy)
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9

        # --- Phase 4: Coordinate-wise line search ---
        if time_left() and calls < budget:
            x_ls = best_x.copy()
            f_ls = best_f
            for coord in range(dim):
                if not time_left() or calls >= budget:
                    break
                for direction in [1, -1]:
                    if not time_left() or calls >= budget:
                        break
                    step = 0.5
                    improved = True
                    while improved and time_left() and calls < budget:
                        improved = False
                        for trial_factor in [1.0, 0.5, 2.0]:
                            if calls >= budget or not time_left():
                                break
                            trial_step = step * trial_factor
                            trial_x = x_ls.copy()
                            trial_x[coord] = np.clip(x_ls[coord] + direction * trial_step, -5, 5)
                            trial_f = func(trial_x)
                            calls += 1
                            if trial_f < f_ls:
                                x_ls = trial_x
                                f_ls = trial_f
                                update_best(x_ls, f_ls)
                                improved = True
                                break
                        if improved:
                            step *= 1.5
                        else:
                            break

        return best_x, best_f

# EVOLVE-BLOCK-END
