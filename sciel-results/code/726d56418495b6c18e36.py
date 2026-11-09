"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Restart (1+1)-ES with global scan for Lunacek."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin (330s wall clock limit)
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if not np.all(np.isfinite(x)):
                return np.inf
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global coarse search (random sampling)
        # Spend ~20% of budget on global search to find good basins
        global_budget = max(20, total_budget // 5)
        second_best_x = None
        second_best_f = np.inf
        for _ in range(global_budget):
            if time.time() > deadline or calls_made >= total_budget:
                break
            x = rng.uniform(-5.0, 5.0, n)
            f = evaluate(x)
            # Track second-best distinct point (for bi-modal landscape)
            if second_best_x is None:
                second_best_x = x.copy()
                second_best_f = f
            elif f < second_best_f and not np.allclose(x, best_x, atol=0.5):
                second_best_x = x.copy()
                second_best_f = f

        # Phase 2: Local refinement using (1+1)-ES from best point
        if best_x is None:
            best_x = np.zeros(n)
            best_f = evaluate(best_x)

        # (1+1)-ES parameters
        sigma = 1.5  # Initial step size for local refinement
        no_improve = 0
        max_no_improve = 60  # Restart if no improvement for this many steps

        def run_es(start_x, start_f, sigma_init, remaining_budget):
            nonlocal best_x, best_f, calls_made
            local_x = start_x.copy()
            local_f = start_f
            local_sigma = sigma_init
            local_no_improve = 0

            while calls_made < total_budget and time.time() < deadline:
                z = rng.standard_normal(n)
                x_candidate = local_x + local_sigma * z
                x_candidate = np.clip(x_candidate, -5.0, 5.0)

                if not np.all(np.isfinite(x_candidate)):
                    local_no_improve += 1
                    if local_no_improve > max_no_improve:
                        local_x = rng.uniform(-5.0, 5.0, n)
                        local_f = evaluate(local_x)
                        local_sigma = 1.5
                        local_no_improve = 0
                    continue

                f_candidate = evaluate(x_candidate)

                if f_candidate < local_f:
                    local_x = x_candidate.copy()
                    local_f = f_candidate
                    local_no_improve = 0
                    local_sigma *= 1.05
                    if local_sigma > 5.0:
                        local_sigma = 5.0
                else:
                    local_no_improve += 1
                    local_sigma *= 0.95
                    if local_sigma < 0.01:
                        local_sigma = 0.01

                if local_no_improve > max_no_improve:
                    jump_size = 0.5
                    perturbation = rng.uniform(-jump_size, jump_size, n)
                    x_perturbed = local_x + perturbation
                    x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                    f_perturbed = evaluate(x_perturbed)
                    if f_perturbed < local_f:
                        local_x = x_perturbed.copy()
                        local_f = f_perturbed
                    local_sigma = 0.5
                    local_no_improve = 0

            if local_f < best_f:
                best_x = local_x.copy()
                best_f = local_f
            return local_f

        # Refine from best point
        run_es(best_x, best_f, 1.5, total_budget - calls_made)

        # If we found a distinct second-best basin, refine from there too
        if (second_best_x is not None and calls_made < total_budget
                and time.time() < deadline and not np.allclose(second_best_x, best_x, atol=1.0)):
            run_es(second_best_x, second_best_f, 1.5, total_budget - calls_made)

        if best_x is None:
            best_x = np.zeros(n)
            if calls_made < total_budget:
                best_f = evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
