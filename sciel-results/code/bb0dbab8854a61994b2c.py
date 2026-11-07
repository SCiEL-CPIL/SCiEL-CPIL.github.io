"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (mu+lambda)-ES with multi-start and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin within 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start exploration
        # Evaluate several random starting points to find a promising basin
        n_starts = min(10, budget // 10)
        best_x = None
        best_f = np.inf
        eval_count = 0

        for i in range(n_starts):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if eval_count >= budget or time.time() > deadline:
                break

        if best_x is None:
            # Fallback: single evaluation at origin
            x = np.zeros(dim)
            best_f = func(x)
            eval_count += 1
            best_x = x.copy()

        # Phase 2: Population-based adaptive step size search
        mu = 4  # number of parents
        lam = 12  # number of offspring
        sigma = 1.0
        x_best = best_x.copy()
        f_best = best_f

        # Initialize population with best point and some perturbations
        pop = [x_best.copy()]
        for i in range(mu - 1):
            x_pert = np.clip(x_best + 0.5 * rng.standard_normal(dim), -5, 5)
            f_pert = func(x_pert)
            eval_count += 1
            pop.append((x_pert, f_pert))
            if f_pert < f_best:
                f_best = f_pert
                x_best = x_pert.copy()
            if eval_count >= budget or time.time() > deadline:
                break

        # Sort population by fitness
        pop = sorted(pop, key=lambda item: item[1] if isinstance(item, tuple) else np.inf)
        # Ensure all elements are tuples
        pop = [(x, func(x)) if not isinstance(x, tuple) else x for x in pop]
        # Re-evaluate if needed - actually let's just keep it simple
        # Reset population properly
        pop = [(x_best, f_best)]
        for i in range(1, mu):
            if eval_count >= budget or time.time() > deadline:
                break
            x_pert = np.clip(x_best + 0.3 * rng.standard_normal(dim), -5, 5)
            f_pert = func(x_pert)
            eval_count += 1
            pop.append((x_pert, f_pert))
            if f_pert < f_best:
                f_best = f_pert
                x_best = x_pert.copy()

        pop = sorted(pop, key=lambda item: item[1])

        # Main evolution loop
        success_count = 0
        window_size = lam + mu
        success_window = []

        while eval_count < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                parent = pop[rng.integers(mu)]
                child = np.clip(parent[0] + sigma * rng.standard_normal(dim), -5, 5)
                f_child = func(child)
                eval_count += 1
                offspring.append((child, f_child))
                if f_child < f_best:
                    f_best = f_child
                    x_best = child.copy()
                if eval_count >= budget or time.time() > deadline:
                    break

            # Combine and select
            combined = pop + offspring
            combined = sorted(combined, key=lambda item: item[1])
            pop = combined[:mu]

            # Adapt step size using 1/5 rule on recent successes
            # Count how many offspring improved over their parent
            for child, f_child in offspring:
                # Find if this child was better than the best parent
                if f_child < pop[0][1]:
                    success_window.append(1)
                else:
                    success_window.append(0)

            if len(success_window) > window_size:
                success_window = success_window[-window_size:]

            if len(success_window) >= 5:
                success_rate = np.mean(success_window)
                if success_rate > 0.3:
                    sigma *= 1.2
                elif success_rate < 0.1:
                    sigma *= 0.8
                sigma = np.clip(sigma, 1e-4, 10.0)

        # Phase 3: Final high-precision polish
        if eval_count < budget and time.time() < deadline - 10:
            sigma_fine = 0.01
            x_polish = x_best.copy()
            f_polish = f_best
            for _ in range(min(50, budget - eval_count)):
                if time.time() > deadline:
                    break
                y = np.clip(x_polish + sigma_fine * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                eval_count += 1
                if fy < f_polish:
                    x_polish, f_polish = y, fy
                    sigma_fine *= 1.1
                else:
                    sigma_fine *= 0.9
            if f_polish < f_best:
                f_best = f_polish
                x_best = x_polish.copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
