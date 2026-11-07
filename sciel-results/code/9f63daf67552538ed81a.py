"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-model CMA-ES with 1+1-CMA-ES local refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 3:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- 1+1-CMA-ES: adapts covariance to landscape rotation ---
        def cma_1plus1(x0, sigma0, max_calls):
            """1+1-CMA-ES with adaptive covariance matrix."""
            nonlocal total_calls, best_x, best_f
            if max_calls < 5:
                return x0, best_f

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_sqrt_inv = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Parameters for 1+1
            cs = (n + 1.3) / (n + 3 + 0.5 * (n + 1.3))
            cc = (4 + 1.0 / n) / (n + 4 + 2.0 / n)
            c1 = 2.0 / ((n + 1.3) ** 2 + 1.0)
            c_sigma = cs / (n + 2)
            hsig = (1 - (1 - cc) ** (1 / cs)) / np.sqrt(1 - (1 - cc) ** (2 / cs))

            best_x_local = x0.copy()
            best_f_local = best_f

            for _ in range(max_calls):
                if total_calls >= budget or time.time() > deadline - 5:
                    break

                z = rng.standard_normal(n)
                y = x + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break

                if fy < best_f_local:
                    best_f_local = fy
                    best_x_local = y.copy()
                    # Success: update mean, sigma, C
                    d = y - x
                    C_sqrt_inv_d = C_sqrt_inv @ d
                    norm_d = np.linalg.norm(C_sqrt_inv_d)

                    if norm_d > 1e-14:
                        p_sigma = (1 - cs) * p_sigma + cs * (norm_d / np.sqrt(2) - 1) * C_sqrt_inv_d / norm_d
                        sigma *= np.exp(c_sigma * (norm_d / np.sqrt(2) - 1))
                        sigma = np.clip(sigma, 1e-12, 5.0)

                        p_c = (1 - cc) * p_c + cc * hsig * C_sqrt_inv_d / np.sqrt(2)
                        C = (1 + c1 * cc) * C + c1 * np.outer(p_c, p_c)

                        # Recompute C_sqrt_inv
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-14)
                        C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                    x = y.copy()
                else:
                    # Failure: shrink sigma slightly
                    sigma *= 0.95
                    sigma = np.clip(sigma, 1e-12, 5.0)

            return best_x_local, best_f_local

        # --- Phase 1: Initial sampling to find diverse starting points ---
        n_initial = min(30, budget // 8)
        structured_points = [
            np.zeros(n),
            np.ones(n) * 3,
            -np.ones(n) * 3,
            np.ones(n) * 1,
            -np.ones(n) * 1,
        ]
        for p in structured_points:
            if total_calls >= budget:
                break
            try_eval(p)

        for i in range(n_initial):
            if total_calls >= budget or time.time() > deadline - 5:
                break
            scale = 3.5 * (1.0 - i / max(1, n_initial)) + 0.5
            x_try = rng.normal(0, scale, n)
            try_eval(x_try)

        if best_x is None:
            return None, np.inf

        # --- Phase 2: Island model with 1+1-CMA-ES ---
        # Create 3 islands, each starting from a different region
        num_islands = 3
        island_starts = []
        # Best point
        island_starts.append(best_x.copy())
        # Two random points
        for _ in range(num_islands - 1):
            x_rand = rng.uniform(-4, 4, n)
            f_rand = try_eval(x_rand)
            if f_rand is not None:
                island_starts.append(x_rand.copy())
            else:
                island_starts.append(best_x.copy())

        # Run 1+1-CMA-ES on each island in parallel (sequential here, but with budget split)
        budget_per_island = max(10, (budget - total_calls) // num_islands)
        for i, x_start in enumerate(island_starts):
            if total_calls >= budget - 5 or time.time() > deadline - 10:
                break
            remaining = budget - total_calls
            alloc = min(budget_per_island, remaining)
            if alloc < 5:
                break
            x_opt, f_opt = cma_1plus1(x_start, 1.0, alloc)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # --- Phase 3: Final polish with coordinate descent ---
        if total_calls < budget and best_x is not None and time.time() < deadline - 10:
            step = 0.1
            for _ in range(4):
                if total_calls >= budget or time.time() > deadline - 5:
                    break
                improved = False
                for coord in range(n):
                    if total_calls >= budget or time.time() > deadline - 5:
                        break
                    for direction in [-1, 1]:
                        x_try = best_x.copy()
                        x_try[coord] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = try_eval(x_try)
                        if f_try is None:
                            break
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                            improved = True
                if not improved:
                    break
                step *= 0.5

            # Final tiny perturbations
            n_final = min(15, budget - total_calls)
            for i in range(n_final):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                perturbation = rng.normal(0, 0.02, n)
                x_try = best_x + perturbation
                f_try = try_eval(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
