"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for Weierstrass f16.
    Combines:
    - Stratified multi-start exploration (current)
    - CMA-ES style covariance adaptation (inspiration 1)
    - Adaptive population with success-rate based sigma control (inspiration 2)
    - Fine-tuning phase with aggressive convergence (current)
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # ================================================================
        # Phase 1: Stratified multi-start exploration
        # Weierstrass has many local minima; broad sampling helps find global basin
        # ================================================================
        n_init = min(35, budget // 10)
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            # Mix of broad, moderate, and focused sampling
            if i < n_init // 4:
                x = rng.uniform(-4.8, 4.8, dim)
            elif i < n_init // 2:
                x = rng.uniform(-3.5, 3.5, dim)
            elif i < 3 * n_init // 4:
                x = rng.uniform(-2.0, 2.0, dim)
            else:
                x = rng.uniform(-0.8, 0.8, dim)
            evaluate(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # ================================================================
        # Phase 2: CMA-ES style population-based local refinement
        # Use adaptive covariance matrix for better exploration of anisotropic basins
        # ================================================================
        x_center = best_x.copy() if best_x is not None else rng.uniform(-1, 1, dim)
        fx_center = best_f

        # CMA-ES parameters
        pop_size = max(6, dim + 1)  # At least dim+1 for covariance estimation
        sigma = 1.2  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        C_alpha = 1.0 / (4 + np.log(dim))  # Learning rate for covariance
        pc = np.zeros(dim)  # Evolution path for covariance
        cs = 1.0 / (1 + pop_size / 2)  # Learning rate for evolution path
        cc = (1 + 2 / (pop_size + 1)) / (pop_size + 2)  # Cumulation parameter

        successes = []
        window_size = 12
        stagnation = 0
        max_stagnation = 25
        last_improvement_call = calls

        # Track for adaptive sigma
        sigma_history = []

        while calls < budget and time.time() < deadline:
            # Check if we should reduce population for fine-tuning
            if calls > budget * 0.7:
                pop_size = max(4, pop_size - 1)

            # Generate population
            improvements = 0
            offspring = []
            offspring_f = []

            for j in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break

                # Sample from multivariate normal with current covariance
                # Use Cholesky decomposition for stable sampling
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    z = rng.standard_normal(dim)
                    y = x_center + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    y = x_center + sigma * rng.standard_normal(dim)

                y = np.clip(y, -5, 5)
                fy = evaluate(y)

                offspring.append(y)
                offspring_f.append(fy)

                if fy < fx_center:
                    improvements += 1
                    fx_center = fy
                    x_center = y.copy()
                    stagnation = 0
                    last_improvement_call = calls
                else:
                    stagnation += 1

            if len(offspring) == 0:
                break

            # Sort offspring by fitness
            sorted_idx = np.argsort(offspring_f)
            half = max(1, len(offspring) // 2)
            selected = [offspring[i] for i in sorted_idx[:half]]
            selected_f = [offspring_f[i] for i in sorted_idx[:half]]

            # Update center as weighted mean of best half
            weights = np.exp(-np.linspace(0, 1, half) * 2)
            weights /= weights.sum()
            x_center = sum(w * x for w, x in zip(weights, selected))
            x_center = np.clip(x_center, -5, 5)

            # Update evolution path
            d = (x_center - (x_center - sigma * rng.standard_normal(dim))) / sigma if sigma > 0 else np.zeros(dim)
            pc = (1 - cs) * pc + cs * d
            pc_norm = np.linalg.norm(pc)

            # Update covariance matrix
            delta_pc = (1.0 + (2 * (pc_norm ** 2 / dim - 1) / (dim + 1)) * cc) if pc_norm < (1.4 + 2 / (dim + 1)) * cs else 1.0
            C = (1 + C_alpha * (2 - C_alpha) * delta_pc - C_alpha) * C
            for x_sel in selected:
                diff = x_sel - x_center
                C += C_alpha * np.outer(diff, diff)

            # Renormalize covariance to maintain trace
            C /= np.trace(C) / dim

            # Update sigma based on success rate
            successes.append(improvements / pop_size)
            if len(successes) > window_size:
                successes = successes[-window_size:]

            if len(successes) >= window_size:
                avg_rate = np.mean(successes)
                if avg_rate > 0.35:
                    sigma *= 1.15
                elif avg_rate < 0.15:
                    sigma *= 0.85
                else:
                    sigma *= 0.98

            sigma = np.clip(sigma, 0.005, 5.0)

            # Stagnation restart
            if stagnation > max_stagnation:
                if sigma < 0.15:
                    # Likely in sharp local min, restart from best with larger sigma
                    x_center = best_x.copy()
                    fx_center = best_f
                    sigma = 2.0
                    C = np.eye(dim)
                    pc = np.zeros(dim)
                    stagnation = 0
                    successes = []
                else:
                    # Try escape jumps
                    escaped = False
                    for _ in range(min(6, budget - calls)):
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_center + sigma * 2.0 * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                        fy = evaluate(y)
                        if fy < fx_center:
                            x_center, fx_center = y, fy
                            stagnation = 0
                            escaped = True
                            break
                    if not escaped:
                        sigma *= 0.6
                        C = np.eye(dim)
                        pc = np.zeros(dim)
                        stagnation = 0
                    successes = []

        # ================================================================
        # Phase 3: Fine-tuning with aggressive convergence
        # Use remaining budget for small-step local search from best point
        # ================================================================
        if calls < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            fine_sigma = 0.05
            fine_window = 8
            fine_successes = []

            remaining = budget - calls
            fine_budget = max(30, min(remaining, 100))

            for _ in range(fine_budget):
                if calls >= budget or time.time() > deadline:
                    break
                y = x + fine_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    fine_sigma *= 1.1
                    fine_successes.append(1)
                else:
                    fine_sigma *= 0.9
                    fine_successes.append(0)

                if len(fine_successes) > fine_window:
                    fine_successes = fine_successes[-fine_window:]

                if len(fine_successes) >= fine_window:
                    rate = sum(fine_successes) / fine_window
                    if rate > 0.5:
                        fine_sigma *= 1.15
                    elif rate < 0.1:
                        fine_sigma *= 0.75

                if fine_sigma < 0.0003:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
