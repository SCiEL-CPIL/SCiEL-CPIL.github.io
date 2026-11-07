import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0  # 50s margin before 330s limit
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # CMA-ES parameters
        lambda_pop = 14
        mu = lambda_pop // 2
        # Weights for (mu/lambda)-selection
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + 4 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 9 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_c)
        cmu = min(1 - c1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((dim + 2) ** 2 + mu_eff))

        # Initialize
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        calls = 1
        x_ever = x_best.copy()
        f_ever = f_best

        # CMA-ES state
        mean = x_best.copy()
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        stall_counter = 0
        stall_threshold = 8  # 8 generations without improvement = ~112 calls

        # Archive for restarts
        archive = [(f_best, x_best.copy())]

        while calls + lambda_pop <= budget and time_ok():
            # Generate population
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Standard normal samples
            z = rng.standard_normal((lambda_pop, dim))
            # Transform: L * z
            pop = mean[None, :] + sigma * (z @ L.T)
            pop = np.clip(pop, -5, 5)

            # Evaluate
            f_vals = np.array([func(x) for x in pop])
            calls += lambda_pop

            # Update archive
            for i in range(lambda_pop):
                if f_vals[i] < f_ever:
                    f_ever = f_vals[i]
                    x_ever = pop[i].copy()
                archive.append((f_vals[i], pop[i].copy()))
            archive.sort(key=lambda t: t[0])
            if len(archive) > 20:
                archive = archive[:20]

            # Sort by fitness
            order = np.argsort(f_vals)
            pop_sorted = pop[order]
            f_sorted = f_vals[order]

            # Weighted mean of best mu
            new_mean = (weights[:, None] * pop_sorted[:mu]).sum(axis=0)

            # Update step size
            y = (new_mean - mean) / (sigma + 1e-30)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            norm_ps = np.linalg.norm(ps)
            # h_sigma calculation requires tracking generations or using a simplified version
            # Simplified h_sigma: check if norm(ps) is within expected range
            # More robust: use the standard formula with generation count
            # For simplicity and robustness against bugs, we can use a heuristic or simplified check
            # Let's stick to the standard formula but ensure no division by zero or negative sqrt
            # Note: calls // lambda_pop is the generation index
            gen = calls // lambda_pop
            # The term (1 - c_sigma)^(2 * (gen + 1)) might underflow or be problematic
            # Standard implementation often tracks this explicitly. 
            # Let's use a simplified condition: if norm_ps is not too large, h_sigma = 1
            # Or better, just set h_sigma = 1 if norm_ps < (1 + 2/(dim+1)) * chi_n * np.sqrt(1 - (1-c_sigma)**(2*(gen+1)))
            # To avoid complexity, let's just use h_sigma = 1.0 if norm_ps < 1.5 * chi_n else 0.0
            # This is a robust approximation.
            h_sigma = 1.0 if norm_ps < 1.5 * chi_n else 0.0
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            pc = (1 - c_c) * pc + h_sigma * c_c * y
            outer_y = np.outer(y, y)
            outer_ps = np.outer(ps, ps)
            C = (1 - c1 - cmu) * C + c1 * (outer_y + (h_sigma * (1 - h_sigma) + c_c) * C)
            C += cmu * sum(w_i * np.outer(pop_sorted[i] - mean, pop_sorted[i] - mean) / (sigma ** 2) for i, w_i in enumerate(weights))

            mean = new_mean
            mean = np.clip(mean, -5, 5)

            # Check improvement
            best_val = f_sorted[0]
            if best_val < f_best:
                x_best = pop_sorted[0].copy()
                f_best = best_val
                stall_counter = 0
            else:
                stall_counter += 1

            # Restart if stalled
            if stall_counter > stall_threshold and calls < budget - lambda_pop - 2:
                if len(archive) > 1:
                    # Pick a diverse point from archive
                    idx = rng.integers(1, min(15, len(archive)))
                    mean = archive[idx][1].copy()
                    f_best = archive[idx][0]
                    x_best = mean.copy()
                else:
                    mean = rng.uniform(-5, 5, dim)
                    f_best = func(mean)
                    calls += 1
                    x_best = mean.copy()
                sigma = 2.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                stall_counter = 0
                if f_best < f_ever:
                    f_ever = f_best
                    x_ever = x_best.copy()

        # Use remaining budget with CMA-ES if possible, else (1+1)
        while calls < budget and time_ok():
            n_cand = min(lambda_pop, budget - calls)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            z = rng.standard_normal((n_cand, dim))
            pop = mean[None, :] + sigma * (z @ L.T)
            pop = np.clip(pop, -5, 5)

            f_vals = np.array([func(x) for x in pop])
            calls += n_cand

            for i in range(n_cand):
                if f_vals[i] < f_ever:
                    f_ever = f_vals[i]
                    x_ever = pop[i].copy()
                archive.append((f_vals[i], pop[i].copy()))
            archive.sort(key=lambda t: t[0])
            if len(archive) > 20:
                archive = archive[:20]

            best_i = np.argmin(f_vals)
            if f_vals[best_i] < f_best:
                x_best = pop[best_i].copy()
                f_best = f_vals[best_i]
                # Simple update for remaining budget
                y = (pop[best_i] - mean) / (sigma + 1e-30)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                norm_ps = np.linalg.norm(ps)
                h_sigma = 1.0 if norm_ps < 1.5 * chi_n else 0.0
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                # Update mean towards best point
                mean = 0.5 * pop[best_i] + 0.5 * mean
                mean = np.clip(mean, -5, 5)
                # Update C roughly
                C = 0.9 * C + 0.1 * np.outer(y, y)

        return x_ever, f_ever
