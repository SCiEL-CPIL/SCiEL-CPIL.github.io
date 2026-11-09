"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse Grid Search + CMA-ES Refinement for BBOB f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def run_cma_es(x0, sigma0, max_calls, max_gen=50):
            """Run CMA-ES from x0 with initial sigma0, using at most max_calls evaluations."""
            nonlocal call_count, best_x, best_f

            x = x0.copy()
            sigma = sigma0

            mu = 4
            lambda_ = 12
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            gen = 0
            stagnation = 0
            max_stagnation = 12

            while (call_count < budget and time.time() < deadline and
                   gen < max_gen and stagnation < max_stagnation):
                if gen % 25 == 0 and gen > 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-12)
                        C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                    except:
                        C_inv_sqrt = np.eye(dim)

                offspring = []
                off_f = []
                for k in range(lambda_):
                    if call_count >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                    fy = call_func(y)
                    if fy is None:
                        break
                    off_f.append(fy)

                if len(off_f) < mu:
                    break

                sorted_idx = np.argsort(off_f)
                selected_x = [offspring[i] for i in sorted_idx[:mu]]
                selected_f = [off_f[i] for i in sorted_idx[:mu]]

                if selected_f[0] < best_f - 1e-10:
                    stagnation = 0
                else:
                    stagnation += 1

                m = sum(w * xk for w, xk in zip(weights, selected_x))

                diff = m - x
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff

                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) / chi_n
                if h_sigma < (1.4 + 2 / (dim + 1)):
                    sigma *= np.exp(c_sigma / chi_n * (h_sigma - 1))

                delta_d = (1 + c_mu * (mueff - 1)) * diff
                rank_mu = np.outer(delta_d, delta_d)
                p_c_norm_sq = np.linalg.norm(p_c)**2
                h_c = 1.0 if p_c_norm_sq > 1 / (1 - c_c)**2 else 0.0
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + h_c * np.eye(dim)) + c_mu * rank_mu

                x = m
                gen += 1

            return x, sigma

        # Phase 1: Coarse Grid Search
        # Rastrigin has period 2.0. A grid with step 4.0 samples every 2nd local minimum.
        # Points: -4, 0, 4. (3 points per dim -> 243 total)
        grid_vals = [-4.0, 0.0, 4.0]
        grid_points = []
        for i in grid_vals:
            for j in grid_vals:
                for k in grid_vals:
                    for l in grid_vals:
                        for m in grid_vals:
                            grid_points.append(np.array([i, j, k, l, m]))
        
        # Shuffle to ensure we don't stop early if the best is late? 
        # Actually, we want to evaluate all if possible.
        # 243 calls is ~50% of budget.
        for p in grid_points:
            if call_count >= budget or time.time() > deadline:
                break
            val = call_func(p)
            if val is None:
                break

        # Phase 2: Refinement with CMA-ES
        # Start from the best point found in the grid
        # Use a moderate sigma to allow escaping local minima if the grid point was not exactly the min
        if call_count < budget - 20 and time.time() < deadline:
            run_cma_es(best_x.copy(), 1.0, budget - call_count, max_gen=100)
            
        # If we have budget left, try a second best point?
        # Or just continue refining the best?
        # CMA-ES is stochastic, so restarting from the best with a smaller sigma might help if it stagnated.
        # But the first run should be sufficient if it converges.
        
        # Fallback if grid didn't run (should not happen)
        if call_count < 10:
            while call_count < budget and time.time() < deadline:
                x = rng.uniform(-5, 5, dim)
                call_func(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
