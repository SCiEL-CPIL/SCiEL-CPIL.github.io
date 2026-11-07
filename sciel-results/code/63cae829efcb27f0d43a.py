"""
Hybrid optimizer for BBOB f19 (Griewank-Rosenbrock).
Combines the robust multi-restart CMA-ES structure from the inspiration
with the adaptive local refinement and initial exploration from the current program.
"""
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES with local polish and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return np.inf
            calls_made += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                if np.isfinite(fx):
                    return float(fx)
                return np.inf
            except Exception:
                return np.inf

        rng = np.random.default_rng(42)

        # --- Phase 1: Initial Exploration ---
        # Diverse starting points to find a good basin for the first CMA-ES run
        init_points = [np.zeros(dim)]
        # Add some structured points
        for val in [-3, 0, 3]:
            init_points.append(np.full(dim, val))
        # Random points
        for _ in range(12):
            init_points.append(rng.uniform(-5, 5, dim))

        for p in init_points:
            if calls_made >= total_budget or time.time() > deadline - 10:
                break
            fx = safe_call(p)
            if fx < best_f:
                best_f = fx
                best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # --- CMA-ES Parameters ---
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # 11 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (2 + mueff / dim) / (dim + 2 + mueff / dim)
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (1 - cc - cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        hsig_threshold = (1.4 + 2 / (dim + 1)) * np.sqrt(dim * (1 - 1 / dim ** 2))

        def run_cmaes_restart(x_start, sigma_start, max_evals):
            """
            Run a single instance of CMA-ES.
            Returns the best x and f found in this run.
            """
            nonlocal calls_made, best_f, best_x, deadline

            if max_evals <= 0 or time.time() > deadline - 5:
                return x_start.copy(), best_f

            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            pc = np.zeros(dim)
            p_sigma = np.zeros(dim)

            best_x_local = x_start.copy()
            best_f_local = best_f
            evals_in_run = 0
            gen = 0
            last_best_f = best_f
            stagnation_counter = 0

            while evals_in_run < max_evals:
                if time.time() > deadline - 5:
                    break
                if calls_made + lambda_ > total_budget:
                    break

                # Stagnation check: if no improvement in 5 generations, stop this restart
                if best_f_local < last_best_f - 1e-6:
                    last_best_f = best_f_local
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter > 5:
                        break

                # Generate offspring
                try:
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C = (C + C.T) / 2
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))

                z = rng.standard_normal((lambda_, dim))
                Y = z @ B.T
                Offspring = x_mean[None, :] + sigma * Y
                Offspring = np.clip(Offspring, -5, 5)

                # Evaluate offspring
                Offspring_F = []
                for i in range(lambda_):
                    if calls_made >= total_budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    fi = safe_call(Offspring[i])
                    Offspring_F.append(fi)
                    evals_in_run += 1
                    if fi < best_f_local:
                        best_f_local = fi
                        best_x_local = Offspring[i].copy()

                if len(Offspring_F) < lambda_:
                    break

                Offspring_F = np.array(Offspring_F)
                order = np.argsort(Offspring_F)
                Offspring = Offspring[order]
                Offspring_F = Offspring_F[order]

                # Update global best
                if Offspring_F[0] < best_f:
                    best_f = Offspring_F[0]
                    best_x = Offspring[0].copy()

                gen += 1

                # Update mean
                x_mean_new = np.sum(weights[:, np.newaxis] * Offspring[:mu], axis=0)

                # Update paths
                y_diff = (x_mean_new - x_mean) / (sigma + 1e-300)
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y_diff
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_diff

                # Sigma adaptation
                ps_norm = np.linalg.norm(p_sigma)
                # Using the standard CMA-ES sigma update rule
                # Note: The denominator term (1 - (1 - cs)^(2*gen)) is for the initial phase,
                # but standard implementations often just use the threshold.
                # We'll stick to the standard robust implementation.
                h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - cs) ** (2 * gen)) < hsig_threshold else 0.0
                sigma *= np.exp((cs / 5) * (ps_norm / np.sqrt(1 - (1 - cs) ** (2 * gen)) - hsig_threshold))

                # Update covariance
                delta_h = 1 + 2 * max(0, ps_norm / np.sqrt(1 - (1 - cs) ** (2 * gen)) - 1) * h_sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cc * (2 - cc)) * delta_h * np.eye(dim))

                # Weighted outer products
                Y_diff = (Offspring[:mu] - x_mean[None, :]) / (sigma + 1e-300)
                C += cmu * np.sum(weights[:, np.newaxis, np.newaxis] * Y_diff[:, :, np.newaxis] * Y_diff[:, np.newaxis, :], axis=0)

                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-12):
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                x_mean = x_mean_new

            return best_x_local, best_f_local

        # --- Phase 2: Multi-Restart CMA-ES ---
        # We allocate the remaining budget to a few restarts.
        # The first restart gets the most budget.
        remaining_budget = total_budget - calls_made
        
        # Strategy:
        # 1. Run CMA-ES from the best found point with a reasonable sigma.
        # 2. If it stagnates, restart from the best found point with a smaller sigma (exploit)
        #    or a slightly perturbed point (explore).
        
        # Let's use 3 restarts.
        # Budget allocation: 50%, 30%, 20% of remaining.
        restarts = [0.5, 0.3, 0.2]
        
        current_x = best_x.copy()
        current_sigma = 1.0

        for frac in restarts:
            if calls_made >= total_budget or time.time() > deadline - 10:
                break
            
            budget_for_this_restart = int(remaining_budget * frac)
            if budget_for_this_restart < lambda_:
                break

            # Adjust sigma for restart
            # First restart: 1.0
            # Second restart: 0.5 (exploit)
            # Third restart: 0.2 (fine-tune)
            if frac > 0.4:
                current_sigma = 1.0
            elif frac > 0.25:
                current_sigma = 0.5
            else:
                current_sigma = 0.2

            new_best_x, new_best_f = run_cmaes_restart(current_x, current_sigma, budget_for_this_restart)
            
            if new_best_f < best_f:
                best_f = new_best_f
                best_x = new_best_x.copy()
            
            current_x = best_x.copy()
            remaining_budget -= budget_for_this_restart

        # --- Phase 3: Final Local Refinement ---
        # Use remaining budget for a simple hill-climbing / random walk polish
        # This helps in the final stages where CMA-ES might be too coarse.
        if best_x is not None and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            local_sigma = 0.3
            # Use up to 20% of total budget or remaining, whichever is smaller
            max_local_calls = min(20, total_budget - calls_made)
            
            for _ in range(max_local_calls):
                if time.time() > deadline - 2:
                    break
                if calls_made >= total_budget:
                    break
                y = np.clip(x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    local_sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    local_sigma *= 0.8
                    if local_sigma < 1e-4:
                        break

        return best_x, best_f
