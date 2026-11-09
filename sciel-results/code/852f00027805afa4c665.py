import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 280  # Safety margin for 300s limit

        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.perf_counter() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")

            x = np.asarray(x, dtype=np.float64)
            f = func(x)
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 0: Multi-point initialization to find a good starting region
        n_init = min(5, self.budget // 20)
        candidates = []
        for i in range(n_init):
            if time.perf_counter() > deadline or calls >= self.budget:
                break
            if i == 0:
                x_c = rng.uniform(-0.5, 0.5, dim)
            else:
                scale = 1.5 + 0.5 * i
                x_c = rng.uniform(-scale, scale, dim)
            x_c = np.clip(x_c, -5.0, 5.0)
            try:
                f_c = safe_func(x_c)
            except (TimeoutError, StopIteration):
                break
            candidates.append((f_c, x_c))

        if not candidates:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        fx, x = candidates[0]

        # Phase 1: CMA-ES with population size 4 and integrated restart rule
        # This handles rotated/scaled landscapes better than (1+1)-ES
        n = dim
        lambda_ = 4  # population size
        mu = lambda_ // 2  # number of parents
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        # Step size
        sigma = 2.0
        sigma_min = 1e-8
        sigma_max = 10.0

        # Covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt_inv = np.eye(n)
        C_sqrt = np.eye(n)

        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Restart conditions
        stall_count = 0
        STALL_LIMIT = 60
        restart_count = 0
        MAX_RESTARTS = 5

        # Track best for restarts
        local_best_x = x.copy()
        local_best_f = fx

        while calls < self.budget and time.perf_counter() < deadline:
            # Generate population
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5.0, 5.0)
                offspring.append(y)

            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                try:
                    fy = safe_func(y)
                except (TimeoutError, StopIteration):
                    break
                offspring_f.append(fy)

            if len(offspring_f) < lambda_:
                break

            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring_sorted = [offspring[i] for i in idx]
            offspring_f_sorted = [offspring_f[i] for i in idx]

            # Update best
            if offspring_f_sorted[0] < local_best_f:
                local_best_f = offspring_f_sorted[0]
                local_best_x = offspring_sorted[0].copy()

            # Mean of best mu offspring
            m = np.zeros(n)
            for i in range(mu):
                m += w[i] * offspring_sorted[i]

            # Evolution path for covariance matrix
            y = (m - x) / sigma
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * y

            # Covariance matrix adaptation
            if np.linalg.norm(pc) ** 2 / (1 - (1 - c_c) ** (2 * (calls - n_init))) <= (1 + 2 * c_c) * n + 5:
                C += c_1 * (np.outer(pc, pc) - C)
                for i in range(mu):
                    C += c_mu * w[i] * (np.outer(y - pc, y - pc) - C)

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C_inv and C_sqrt_inv
            C_inv = np.linalg.inv(C)
            C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
            C_sqrt = np.linalg.cholesky(C)

            # Step size adaptation (CSA)
            h_sigma = (np.linalg.norm(C_sqrt_inv @ pc) / np.sqrt(1 - (1 - c_c) ** (2 * (calls - n_init)))) / (np.sqrt(n) * (1 - c_c / 2)) < (1.4 + 2 / (n + 1))
            ps = (1 - c_sigma) * ps + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ y)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - n_init))) - np.sqrt(n)))

            sigma = max(sigma_min, min(sigma, sigma_max))

            # Update mean
            x = m

            # Stall detection
            if local_best_f <= best_f:
                stall_count = 0
            else:
                stall_count += 1

            if stall_count >= STALL_LIMIT and restart_count < MAX_RESTARTS:
                # Restart from a new random point
                x_new = rng.uniform(-3.0, 3.0, n)
                x_new = np.clip(x_new, -5.0, 5.0)
                try:
                    fx_new = safe_func(x_new)
                except (TimeoutError, StopIteration):
                    break
                if fx_new < local_best_f:
                    x = x_new
                    local_best_f = fx_new
                    local_best_x = x_new.copy()
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    C_sqrt_inv = np.eye(n)
                    C_sqrt = np.eye(n)
                    ps = np.zeros(n)
                    pc = np.zeros(n)
                    sigma = 2.0
                stall_count = 0
                restart_count += 1

        # Phase 2: Nelder-Mead local refinement
        remaining = self.budget - calls
        if remaining > 10 and best_x is not None:
            try:
                result = scipy.optimize.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass

        # Phase 3: Adaptive coordinate-descent polish
        remaining = self.budget - calls
        if remaining > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.05
            max_cd_rounds = min(5, remaining // (2 * dim))

            for _round in range(max_cd_rounds):
                if time.perf_counter() > deadline or calls >= self.budget:
                    break
                moved_any = False
                for d in range(dim):
                    if calls >= self.budget or time.perf_counter() > deadline:
                        break
                    best_trial = None
                    best_trial_f = fx
                    for sign in (1.0, -1.0):
                        if calls >= self.budget:
                            break
                        trial = x.copy()
                        trial[d] = np.clip(trial[d] + sign * step, -5, 5)
                        try:
                            ftrial = safe_func(trial)
                        except (TimeoutError, StopIteration):
                            return best_x, best_f
                        if ftrial < best_trial_f:
                            best_trial_f = ftrial
                            best_trial = trial.copy()
                    if best_trial is not None:
                        x = best_trial
                        fx = best_trial_f
                        moved_any = True

                if not moved_any:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
