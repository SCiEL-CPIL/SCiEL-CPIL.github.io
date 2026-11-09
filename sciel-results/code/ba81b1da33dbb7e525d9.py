# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """
    Hybrid optimizer for BBOB f11 (Discus).
    
    Strategy:
    1. Quick random initialization to identify the ring structure.
    2. Differential Evolution to break symmetry and find the basin.
    3. CMA-ES for precise local convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 275  # Safety margin for 300s limit

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(42)

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

        # Phase 1: Rapid exploration to find the ring center
        # Discus function has a ring of local minima. We need to find a point on the ring
        # and then move towards the center where the global minimum is.
        
        # Initial random points to estimate the ring radius
        n_init = min(10, budget)
        for i in range(n_init):
            if timed_out() or call_count >= budget:
                break
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = 4.0 * rng.standard_normal(dim)
            evaluate(x0)
        
        if best_x is None:
            return None, np.inf

        # Phase 2: Differential Evolution to escape local minima and approach the global basin
        # DE is robust and good at exploring multi-modal landscapes.
        remaining_budget = budget - call_count
        if remaining_budget > 50 and not timed_out():
            # DE bounds
            bounds = [(-5.0, 5.0)] * dim
            
            def de_func(x):
                nonlocal call_count
                if call_count >= budget or timed_out():
                    return 1e10
                return evaluate(x)

            try:
                # Use a small population to save budget, but enough to explore
                # maxiter is set high but will stop due to budget/deadline
                result = differential_evolution(
                    de_func,
                    bounds,
                    seed=42,
                    maxiter=50,
                    popsize=10,
                    tol=1e-8,
                    polish=False,  # We will do our own polish
                    init='sobol'
                )
                if result.fun < best_f:
                    best_f = float(result.fun)
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
            except Exception:
                pass

        # Phase 3: CMA-ES for fine-tuning
        if call_count < budget and best_x is not None and not timed_out():
            # CMA-ES setup
            lambda_ = 10
            mu_ = 5
            w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1.0) + c_sigma
            c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
            c_1 = 2 / (dim ** 2 + 6)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x_mean = best_x.copy()
            sigma = 0.5  # Start with smaller sigma for local refinement
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            L = np.eye(dim)
            have_L = True

            while call_count < budget:
                if timed_out():
                    break

                offspring = []
                for _ in range(lambda_):
                    if call_count >= budget or timed_out():
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    f = evaluate(y)
                    offspring.append((f, y, z))

                if len(offspring) < mu_:
                    break

                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu_]

                old_mean = x_mean.copy()
                x_mean = np.sum([w[i] * selected[i][1] for i in range(mu_)], axis=0)
                delta = x_mean - old_mean

                if have_L:
                    try:
                        Cinv_half_delta = np.linalg.solve(L.T, np.linalg.solve(L, delta))
                    except np.linalg.LinAlgError:
                        have_L = False
                        Cinv_half_delta = delta
                else:
                    Cinv_half_delta = delta

                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * Cinv_half_delta
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * \
                    np.sum([w[i] * selected[i][2] for i in range(mu_)], axis=0)

                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = float(np.clip(sigma, 1e-10, 5.0))

                drift = 1 - np.exp(-2 * c_1 * np.dot(p_c, p_c))
                C = (1 - c_1 - c_mu) * (L @ L.T)
                C += c_1 * (drift * np.outer(p_c, p_c) + (1 - np.exp(-2 * c_sigma)) * np.eye(dim))
                for i in range(mu_):
                    d = selected[i][1] - x_mean
                    C += c_mu * w[i] * np.outer(d, d)
                C = 0.5 * (C + C.T)

                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                    have_L = True
                except np.linalg.LinAlgError:
                    L = L + 1e-4 * np.eye(dim)
                    have_L = False

                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
