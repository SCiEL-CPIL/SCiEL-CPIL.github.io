"""Combined optimizer: Sobol-sampled integer basin selection + CMA-ES + (1+1)-ES fine-tuning for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol-based periodic basin selection followed by CMA-ES with (1+1)-ES fine-tuning."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        def time_ok():
            return time.time() < deadline and calls[0] < budget

        # ---------- Phase 1: Sobol-sampled Integer Basin Selection ----------
        # Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)).
        # Local minima are at integer points. The global minimum is at a shifted integer point s.
        # At any integer point k, f(k) = 10*n + sum(k_i^2 - 10) = 10*n - 10*n + sum(k_i^2) - 10*sum(1)?? 
        # Wait, f(k) = 10*n + sum(k_i^2 - 10*cos(2*pi*k_i)). Since cos(2*pi*integer)=1,
        # f(k) = 10*n + sum(k_i^2 - 10) = sum(k_i^2) - 10*n + 10*n = sum(k_i^2)? No.
        # f(k) = 10*n + sum(k_i^2) - 10*sum(1) = sum(k_i^2). 
        # Actually, the offset is usually such that f(0) = 10*n. 
        # Let's check: f(0) = 10*n + 0 - 10*n = 0? No, standard Rastrigin is 10*n + sum(x^2 - 10cos(2pi x)).
        # f(0) = 10*n - 10*n = 0.
        # If shifted by s, f(x) = 10*n + sum((x-s)^2 - 10cos(2pi(x-s))).
        # At x=k (integer), x-s is not necessarily integer.
        # However, the problem states "hidden seeded shift... minimizer lies in [-4, 4]^5".
        # The local minima of the shifted function are at s + integer_vector.
        # Since s is in [-4, 4]^5, the local minima are in [-9, 9]^5.
        # But the domain is [-5, 5]^5.
        # So we only care about local minima within the box.
        # The value at a local minimum m is f(m) = 10*n (if m is the global min) or higher.
        # Actually, f(m) = 10*n + sum((m-s)^2 - 10). If m = s + k, then m-s=k.
        # f(m) = 10*n + sum(k_i^2 - 10) = sum(k_i^2).
        # So the function value at a local minimum is the squared distance from the global minimum (shift s).
        # We want to find the integer point k (relative to s) that is 0, i.e., find s.
        # We can sample integer points in the domain [-5, 5]^5.
        # The candidates for the global minimum are the integer points in [-4, 4]^5 (since s is there).
        # We use a Sobol sequence to sample these integer points efficiently.

        best_x = None
        best_f = np.inf
        
        # Number of integer candidates to evaluate
        # 9^5 = 59049 total. We can afford ~50-100 evaluations here.
        n_explore = 60
        
        # Generate Sobol points
        sobol_engine = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sobol_engine.random(n_explore)
        
        # Map Sobol points [0,1)^5 to integer coordinates in [-4, 4]^5
        # There are 9 integers: -4, -3, ..., 4
        # We can map u in [0,1) to index 0..8
        int_candidates = []
        for pt in sobol_points:
            # pt is in [0,1)
            # Scale to 0..8.999, floor to 0..8
            idx = np.floor(pt * 9).astype(int)
            # Map index 0..8 to -4..4
            coords = idx - 4
            int_candidates.append(coords.astype(float))
            
        # Evaluate these integer points
        for x_cand in int_candidates:
            if not time_ok():
                break
            x_cand = np.clip(x_cand, -5, 5)
            fx = call_func(x_cand)
            if fx < best_f:
                best_f = fx
                best_x = x_cand.copy()
                
        # If we didn't find any (shouldn't happen), fallback to center
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = call_func(best_x)

        # ---------- Phase 2: CMA-ES ----------
        # Since we are likely at or very near the global minimum (an integer point),
        # we can use a small sigma to converge quickly.
        
        def cma_es(x0, f0, budget_cma, sigma_init=0.5):
            dim_local = self.dim
            pop_size = 4 + int(3 * np.log(dim_local))
            mu = pop_size // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            cc = (4 + dim_local / dim_local) / (dim_local + 4)
            cs = (4 + dim_local / 3) / (dim_local + 4)
            c1 = (2 + 2 / (dim_local + 1)) / ((dim_local + 2) ** 2 + 2)
            cmu = min(1 - c1,
                      2 * (mu_eff - 2 + 1 / mu_eff) / ((dim_local + 2) ** 2 + mu_eff))

            sigma = sigma_init
            x = x0.copy()
            f_best = f0
            C = np.eye(dim_local)
            C_sqrt = np.eye(dim_local)
            C_sqrt_inv = np.eye(dim_local)
            ps = np.zeros(dim_local)
            pc = np.zeros(dim_local)

            eigvals = np.ones(dim_local)
            eigvecs = np.eye(dim_local)

            for gen in range(budget_cma // pop_size + 1):
                if not time_ok():
                    break

                # Refresh eigendecomposition periodically
                if gen % 2 == 0 or gen == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                    C_sqrt_inv = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T

                # Generate population
                offspring = []
                for i in range(pop_size):
                    if not time_ok():
                        break
                    z = rng.standard_normal(dim_local)
                    y = np.clip(x + sigma * (C_sqrt @ z), -5, 5)
                    fy = call_func(y)
                    offspring.append((fy, y))

                if not offspring:
                    break

                offspring.sort(key=lambda t: t[0])

                # Update best
                for fy, y in offspring:
                    if fy < f_best:
                        f_best = fy
                        x = y.copy()
                        break

                # Weighted recombination
                x_mean = np.zeros(dim_local)
                for i in range(mu):
                    x_mean += w[i] * offspring[i][1]

                # Step size adaptation (1/5 rule via path ps)
                d_sigma = 1 + 2 * max(0, (cc + 1) / (cc + dim_local / 2) - 1)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (
                    C_sqrt_inv @ (x_mean - x)) / sigma

                hsig = (np.linalg.norm(ps) /
                        np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))))
                if hsig < (1.4 + 2 / (dim_local + 1)) * np.sqrt(mu_eff + 1):
                    sigma *= np.exp(cs / d_sigma * (
                        np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))

                # Evolution path for covariance
                pc = (1 - cc) * pc + cc * (x_mean - x) / (
                    sigma * np.sqrt(cc * (2 - cc) * mu_eff))

                # Covariance update
                delta_sigma = (np.linalg.norm(pc) /
                               np.sqrt(1 - (1 - cc) ** (gen + 1)) >
                               1.4 + 2 / (dim_local + 1))
                rank_c = np.outer(pc, pc)
                if delta_sigma:
                    rank_c += (1 / cc - 1) * (2 - cc) * np.eye(dim_local)

                H = np.zeros((dim_local, dim_local))
                for i in range(mu):
                    H += w[i] * np.outer(
                        offspring[i][1] - x, offspring[i][1] - x)

                C = (1 + c1 * (rank_c - np.eye(dim_local)) + cmu * H) * C
                C = (C + C.T) / 2

                # Regularize for positive definiteness
                eigvals_reg, eigvecs_reg = np.linalg.eigh(C)
                eigvals_reg = np.maximum(eigvals_reg, 1e-10)
                C = (eigvecs_reg * eigvals_reg) @ eigvecs_reg.T

                x = x_mean

            return x, f_best

        # First CMA-ES run with the majority of remaining budget
        remaining = budget - calls[0]
        cma_budget_1 = int(remaining * 0.8)
        if cma_budget_1 > 20:
            best_x, best_f = cma_es(best_x, best_f, cma_budget_1, sigma_init=0.5)

        # ---------- Phase 3: (1+1)-ES fine-tuning ----------
        # Use remaining budget for fine-tuning
        if time_ok():
            sigma = 0.1
            # Use remaining calls for (1+1)-ES
            for _ in range(budget - calls[0]):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5, 5)
                fy = call_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.92
                if sigma < 0.001:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
