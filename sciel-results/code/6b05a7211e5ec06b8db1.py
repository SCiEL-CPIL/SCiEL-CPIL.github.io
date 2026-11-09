"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer for Büche-Rastrigin f4: coarse sampling + L-BFGS-B + CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def run_lbfgsb(x0, max_calls):
            """Run L-BFGS-B from x0, using remaining budget."""
            nonlocal calls_made, best_x, best_f

            remaining = total_budget - calls_made
            if remaining < 10:
                return

            # Use a simple L-BFGS-B optimization
            # We'll track function evaluations via a wrapper
            local_calls = [0]
            max_evals = min(remaining, 200)

            def obj(x):
                local_calls[0] += 1
                if local_calls[0] > max_evals or check_deadline():
                    return np.inf
                return evaluate(x)

            try:
                res = minimize(
                    obj, x0, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxfun': max_evals}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        def run_cmaes(x_init, sigma_init, max_calls):
            """CMA-ES with smaller population for faster exploration."""
            nonlocal calls_made, best_x, best_f

            lambda_pop = 8
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)

            cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
            cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
            c1 = 2 / ((lambda_pop + 2) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2) ** 2 + mu_eff))

            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            local_best_f = evaluate(x_mean)
            local_best_x = x_mean.copy()
            no_improve_gen = 0

            while not check_deadline() and calls_made < max_calls:
                offspring = []
                for i in range(lambda_pop):
                    if check_deadline() or calls_made >= max_calls:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = evaluate(x_new)
                    offspring.append((x_new, f_new))

                if len(offspring) == 0:
                    break

                offspring.sort(key=lambda x: x[1])

                x_new_mean = np.sum(
                    [weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0
                )

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (
                    B @ (D * (B.T @ (x_new_mean - x_mean)))
                )

                hsig = np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt(
                    lambda_pop + 1 - mu_eff
                )

                if hsig:
                    sigma *= np.exp(
                        cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1)
                    )

                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (
                    x_new_mean - x_mean
                ) / sigma

                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = (offspring[i][0] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2
                x_mean = x_new_mean

                if offspring[0][1] < local_best_f:
                    local_best_f = offspring[0][1]
                    local_best_x = offspring[0][0].copy()
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1

                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-12))
                except Exception:
                    D2, B = np.linalg.eigh(C + 1e-10 * np.eye(dim))
                    D = np.sqrt(np.maximum(D2, 1e-12))

                if no_improve_gen >= 15 and calls_made < max_calls - 10:
                    break

                if sigma < 1e-14:
                    break

            return local_best_f, local_best_x, calls_made

        # Phase 1: Structured coarse sampling to locate promising basins
        # Büche-Rastrigin has periodic structure with period ~2 in each dimension
        # So we sample at multiples of 2 plus some offsets
        
        # First, evaluate at structured points
        # The function has local minima roughly at integer coordinates
        # Sample a grid of points at even coordinates and midpoints
        
        # Key points: center, corners of the box, and structured grid
        key_points = []
        
        # Origin and symmetric points
        key_points.append(np.zeros(dim))
        
        # Points at ±2 in each coordinate (period of Rastrigin-like functions)
        for sign in [-1, 1]:
            for k in range(dim):
                pt = np.zeros(dim)
                pt[k] = sign * 2.0
                key_points.append(pt)
        
        # Points at ±1 in each coordinate
        for sign in [-1, 1]:
            for k in range(dim):
                pt = np.zeros(dim)
                pt[k] = sign * 1.0
                key_points.append(pt)
        
        # Mixed-sign points
        for combo in [[2, -2, 2, -2, 2], [-2, 2, -2, 2, -2], [1, -1, 1, -1, 1], [-1, 1, -1, 1, -1]]:
            key_points.append(np.array(combo, dtype=np.float64))
        
        # More structured points
        for r in [0.5, 1.5, 2.5, 3.5]:
            for s in [-1, 1]:
                pt = np.full(dim, s * r)
                key_points.append(pt)
        
        # Deduplicate and clip
        unique_points = []
        seen = set()
        for pt in key_points:
            pt_clipped = np.clip(pt, -5, 5)
            key = tuple(np.round(pt_clipped, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(pt_clipped)
        
        for pt in unique_points:
            if check_deadline():
                break
            evaluate(pt)
        
        # Phase 2: Random sampling to explore more of the space
        random_budget = min(150, total_budget - calls_made)
        for _ in range(random_budget):
            if check_deadline():
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)
        
        # Phase 3: Local refinement using L-BFGS-B from the best point
        if not check_deadline() and best_x is not None:
            run_lbfgsb(best_x.copy(), total_budget)
        
        # Phase 4: CMA-ES refinement around the best point
        if not check_deadline() and best_x is not None:
            x_init = best_x.copy()
            sigma_init = 0.8
            max_calls = total_budget
            _, _, _ = run_cmaes(x_init, sigma_init, max_calls)
        
        # Phase 5: If we have budget left, try a few more L-BFGS-B from nearby points
        if not check_deadline() and best_x is not None:
            # Try perturbing the best point and running L-BFGS-B
            for _ in range(3):
                if check_deadline():
                    break
                perturbation = rng.normal(0, 0.5, dim)
                x_try = np.clip(best_x + perturbation, -5, 5)
                run_lbfgsb(x_try, total_budget)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
