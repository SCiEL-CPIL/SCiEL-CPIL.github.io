"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with aggressive restarts for multimodal BBOB f21 landscape.
    Uses covariance matrix adaptation to model correlated peak structures,
    with frequent restarts to escape local minima in the 101-peak landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() > deadline - 5:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_ = 10  # population size
        mu = lambda_ // 2  # number of parents
        # Weights: log-sum weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / (w @ w)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        restart_threshold = 20  # restart after this many steps without improvement
        max_evals_per_run = 50  # max evaluations per CMA-ES run before forced restart

        # Phase 1: Initial diverse sampling
        n_starts = max(5, min(15, budget // 10))
        for i in range(n_starts):
            if not check_time() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is None:
                break

        if best_x is None:
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with aggressive restarts
        if best_x is not None and check_time() and calls_made < budget:
            # Initial CMA-ES run from best point
            while check_time() and calls_made < budget:
                # Initialize CMA-ES state
                mean = best_x.copy()
                sigma = 2.0  # initial step size
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

                no_improve = 0
                evals_in_run = 0
                run_best_f = best_f
                run_best_x = best_x.copy()

                while check_time() and calls_made < budget and evals_in_run < max_evals_per_run:
                    # Generate population
                    offspring = []
                    for i in range(lambda_):
                        # Cholesky decomposition for sampling
                        try:
                            L = np.linalg.cholesky(C)
                        except np.linalg.LinAlgError:
                            L = np.eye(dim)
                        z = rng.standard_normal(dim)
                        x_i = mean + sigma * (L @ z)
                        x_i = np.clip(x_i, -5, 5)
                        offspring.append(x_i)

                    # Evaluate offspring
                    f_offspring = []
                    for x_i in offspring:
                        f_i = safe_call(x_i)
                        if f_i is None:
                            f_offspring.append(np.inf)
                        else:
                            f_offspring.append(f_i)
                        evals_in_run += 1
                        if f_i < run_best_f:
                            run_best_f = f_i
                            run_best_x = x_i.copy()

                    if not check_time() or calls_made >= budget:
                        break

                    # Sort by fitness
                    indices = np.argsort(f_offspring)
                    y_sorted = [offspring[i] for i in indices]
                    f_sorted = [f_offspring[i] for i in indices]

                    # Update mean
                    y_weighted = np.zeros(dim)
                    for i in range(mu):
                        y_weighted += w[i] * (y_sorted[i] - mean) / sigma
                    mean_new = mean + sigma * y_weighted
                    mean_new = np.clip(mean_new, -5, 5)

                    # Update step-size control (CSA)
                    h_sigma = np.linalg.norm(y_weighted) / (1 - (1 - c_sigma) ** (1 / c_sigma)) / chi_n
                    if h_sigma < 1.4:
                        p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) / chi_n * y_weighted
                    sigma *= np.exp(c_sigma / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
                    sigma = max(sigma, 1e-8)
                    sigma = min(sigma, 10.0)

                    # Update covariance matrix
                    h_c = np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (1 / c_c)) / chi_n
                    if h_c < (1 + 2 / (mu_eff + 1)) * 1.4:
                        p_c = (1 - c_c) * p_c + c_c * (np.linalg.norm(mean_new - mean) > 0) * (mean_new - mean) / sigma

                    # Rank-one update
                    y_diff = (mean_new - mean) / sigma
                    C = (1 - c_1 - c_mu) * C + c_1 * (2 - c_1) * C + c_1 * np.outer(y_diff, y_diff) + c_mu * sum(w[i] * np.outer((y_sorted[i] - mean) / sigma, (y_sorted[i] - mean) / sigma) for i in range(mu))

                    # Ensure positive semi-definiteness
                    C = 0.5 * (C + C.T)
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C_sqrt = np.eye(dim)

                    mean = mean_new

                    # Check for improvement
                    if run_best_f < best_f - 1e-10:
                        best_f = run_best_f
                        best_x = run_best_x.copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                    # Aggressive restart condition
                    if no_improve >= restart_threshold:
                        break

                # After each CMA-ES run, decide whether to restart
                # Restart from best found so far with fresh CMA-ES state
                # Occasionally do a global jump to explore new regions
                if check_time() and calls_made < budget:
                    if no_improve >= restart_threshold and rng.random() < 0.3:
                        # Global restart: jump to a new random location
                        jump = rng.uniform(-5, 5, dim)
                        jump = np.clip(jump, -5, 5)
                        f_jump = safe_call(jump)
                        if f_jump is not None:
                            if f_jump < best_f:
                                best_f = f_jump
                                best_x = jump.copy()
                            # Start next CMA-ES from this new location
                            mean = jump.copy()
                        else:
                            mean = best_x.copy()
                    else:
                        # Local restart: start from current best
                        mean = best_x.copy()

        # Phase 3: Final local refinement with (1+1)-ES
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            max_local_steps = min(100, budget - calls_made)
            steps = 0

            while check_time() and calls_made < budget and steps < max_local_steps:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                steps += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.95

                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
