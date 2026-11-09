"""CMA-ES with restarts and local polish for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin

        dim = self.dim
        budget = self.budget
        calls_made = 0

        def make_call(x_eval):
            nonlocal calls_made
            x_eval = np.clip(x_eval, -5.0, 5.0)
            f_val = func(x_eval)
            calls_made += 1
            return f_val, x_eval

        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 0: Initial random evaluations to get a good starting point
        rng_init = np.random.default_rng(0)
        for _ in range(min(5, budget)):
            x_rand = rng_init.uniform(-4.0, 4.0, dim)
            f_rand, x_eval = make_call(x_rand)
            update_best(x_eval, f_rand)

        if calls_made >= budget:
            return best_x, best_f

        # Phase 1: CMA-ES with restarts
        # We'll do up to 2 CMA-ES runs. First uses best_x as center, second uses a random point.
        for run_idx in range(2):
            if calls_made >= budget - 80:  # Reserve 80 calls for local polish
                break
            if time.time() > deadline:
                break

            if run_idx == 0:
                x_center = best_x.copy()
                sigma_init = 1.5
                seed = 42
            else:
                # Second restart: random center
                rng_restart = np.random.default_rng(123 + run_idx)
                x_center = rng_restart.uniform(-3.0, 3.0, dim)
                sigma_init = 2.0
                seed = 777 + run_idx

            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(dim)))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            rng = np.random.default_rng(seed)
            x = x_center.copy()
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Track stagnation
            last_improvement_gen = 0
            f_at_last_improvement = best_f
            stagnation_threshold = 15  # generations without improvement

            for gen in range(1, 200):
                if calls_made >= budget - 80:
                    break
                if time.time() > deadline:
                    break

                remaining = budget - 80 - calls_made
                n_gen = min(lambda_, remaining)
                if n_gen < 2:
                    break

                pops = []
                fitnesses = []

                for i in range(n_gen):
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = x + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new, x_eval = make_call(x_new)
                    pops.append(x_eval)
                    fitnesses.append(f_new)
                    update_best(x_eval, f_new)

                if calls_made >= budget - 80:
                    break

                idx = np.argsort(fitnesses)
                pops_sorted = [pops[i] for i in idx]
                fitnesses_sorted = [fitnesses[i] for i in idx]

                w = weights[:len(pops_sorted)]
                if len(w) < mu:
                    w = np.ones(len(pops_sorted)) / len(pops_sorted)

                m = np.sum([w[i] * pops_sorted[i] for i in range(len(w))], axis=0)

                # Update paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B @ (D * (m - x))
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (len(w) + 1))) < (1.4 + 2 / (dim + 1)) * chi_n)
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (m - x)

                # Update C
                rank1 = np.outer(p_c, p_c)
                rankmu = np.zeros((dim, dim))
                for i in range(len(w)):
                    y_i = (pops_sorted[i] - x) / sigma
                    rankmu += w[i] * np.outer(y_i, y_i)

                C = (1 + c1 * h_sigma * (1 - h_sigma) - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

                # Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                B = eigvecs
                D = np.sqrt(eigvals)

                x = m
                sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                if sigma < 1e-12:
                    break

                # Check for stagnation
                if fitnesses_sorted[0] < f_at_last_improvement - 1e-8:
                    f_at_last_improvement = fitnesses_sorted[0]
                    last_improvement_gen = gen
                elif gen - last_improvement_gen > stagnation_threshold:
                    break

        # Phase 2: Local polish using Nelder-Mead
        if best_x is not None and calls_made < budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize

                # Use Nelder-Mead for local refinement
                # Start from best_x
                x0 = best_x.copy()
                f0 = best_f

                # Define objective that counts calls
                def obj_local(x):
                    nonlocal calls_made
                    if calls_made >= budget:
                        return f0
                    x_clipped = np.clip(x, -5.0, 5.0)
                    f_val = func(x_clipped)
                    calls_made += 1
                    if f_val < f0:
                        f0 = f_val
                    return f_val

                # Run Nelder-Mead with limited iterations
                result = minimize(obj_local, x0, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                           'adaptive': True})

                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
