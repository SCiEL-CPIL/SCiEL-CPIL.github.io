"""Multi-restart (1+1)-ES with local polish for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and local polish for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used_calls = 0

        # Phase 1: Global Exploration via Multi-Restart (1+1)-ES
        # Katsuura is multimodal; explore multiple basins
        global_budget = int(budget * 0.85)
        local_budget = budget - global_budget

        # Number of restarts: balance exploration vs. exploitation per restart
        num_restarts = 12
        calls_per_restart = global_budget // num_restarts
        if calls_per_restart < 15:
            num_restarts = max(1, global_budget // 15)
            calls_per_restart = global_budget // num_restarts

        for r in range(num_restarts):
            if used_calls >= global_budget:
                break
            if time.time() > deadline:
                break

            # Random start for this restart
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            local_improvements = 0
            max_no_improve = 20  # Early termination if stuck

            for _ in range(calls_per_restart):
                if used_calls >= global_budget:
                    break
                if time.time() > deadline:
                    break

                # Generate offspring
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    local_improvements += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Adaptive sigma: increase on success
                    sigma *= 1.1
                else:
                    local_improvements = 0
                    # Adaptive sigma: decrease on failure
                    sigma *= 0.8

                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 1e-4, 10.0)

                # Early termination if converged locally
                if sigma < 1e-3 or local_improvements == 0 and sigma < 0.05:
                    break

        # Phase 2: Local Polish around the best point found
        if best_x is not None and used_calls < budget:
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls

            # Multi-scale local refinement with decreasing step sizes
            step_sizes = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
            calls_in_phase = 0
            max_calls_in_phase = remaining

            for ss in step_sizes:
                if calls_in_phase >= max_calls_in_phase:
                    break
                if time.time() > deadline:
                    break

                # Allocate calls to this step size proportionally
                calls_for_size = max(1, max_calls_in_phase // len(step_sizes))
                improved_this_size = False

                for _ in range(calls_for_size):
                    if calls_in_phase >= max_calls_in_phase:
                        break
                    if time.time() > deadline:
                        break

                    z = rng.standard_normal(d)
                    y = x + ss * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_in_phase += 1
                    used_calls += 1

                    if fy < fx:
                        x = y
                        fx = fy
                        improved_this_size = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

                # If no improvement at this scale, try next smaller scale
                # (already handled by the loop structure)

        return best_x, best_f
# EVOLVE-BLOCK-END
