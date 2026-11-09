"""Rewritten program for BBOB f7 (Step ellipsoidal) using Simulated Annealing with adaptive cooling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with adaptive cooling for BBOB f7 step ellipsoidal."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize with a broad scout
        x_best = np.clip(rng.uniform(-5, 5, dim), -5, 5)
        f_best = func(x_best)
        calls = 1

        # Simulated Annealing parameters
        T_initial = 50.0  # Initial temperature (large enough to explore)
        T_min = 1e-10
        alpha = 0.995     # Cooling rate
        step_scale = 2.0  # Initial step size scale

        # Adaptive step size tracking
        accept_count = 0
        total_count = 0
        stagnation_window = 50
        stagnation_count = 0
        last_improvement_call = calls

        # Track best value for adaptive step sizing
        f_window_start = f_best

        while calls < budget and time.time() < deadline:
            # Adaptive cooling: slow down cooling when stuck on plateau
            # (no improvement for a while)
            if stagnation_count > stagnation_window:
                # Plateau detected: reduce cooling rate to stay at higher T longer
                alpha_eff = alpha + 0.005  # Slower cooling
            else:
                alpha_eff = alpha

            # Adaptive step size: increase if too many rejections (stuck), decrease if exploring too much
            if total_count > 0:
                accept_rate = accept_count / total_count
                if accept_rate < 0.1:
                    # Too many rejections - increase step to escape local region
                    step_scale *= 1.2
                elif accept_rate > 0.8:
                    # Too many acceptances - decrease step for finer search
                    step_scale *= 0.85
                step_scale = np.clip(step_scale, 0.05, 5.0)

            # Generate candidate move
            x_curr = x_best.copy()
            # Mix of large jumps (to escape plateaus) and small steps (for convergence)
            if rng.random() < 0.3:
                # Large jump: escape potential plateau
                z = rng.standard_normal(dim)
                x_cand = x_curr + step_scale * 2.0 * z
            else:
                # Small step: fine-tuning
                z = rng.standard_normal(dim)
                x_cand = x_curr + step_scale * 0.5 * z

            x_cand = np.clip(x_cand, -5, 5)
            f_cand = func(x_cand)
            calls += 1
            total_count += 1

            # Acceptance criterion
            if f_cand < f_best:
                # Better solution: always accept
                x_best = x_cand.copy()
                f_best = f_cand
                accept_count += 1
                stagnation_count = 0
                last_improvement_call = calls
                f_window_start = f_best
            else:
                # Worse or equal: accept with probability exp(-delta/T)
                delta = f_cand - f_best
                if delta < 1e-8:
                    # Essentially equal - accept to explore flat regions
                    accept_count += 1
                    x_best = x_cand.copy()
                else:
                    T = max(T_min, T_initial * (alpha_eff ** (calls - 1)))
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        accept_count += 1
                        x_best = x_cand.copy()
                        # Don't update f_best since it's worse
                    else:
                        # Reject - stay at current best
                        pass

                stagnation_count += 1

            # Reset counters periodically for adaptive step size
            if total_count >= 100:
                accept_count = 0
                total_count = 0

            # Update temperature
            T_current = max(T_min, T_initial * (alpha_eff ** (calls - 1)))
            if T_current <= T_min * 2:
                # Temperature very low: restart with new random point if budget allows
                if calls < budget - 10 and time.time() < deadline:
                    x_restart = np.clip(rng.uniform(-4, 4, dim), -5, 5)
                    f_restart = func(x_restart)
                    calls += 1
                    if f_restart < f_best:
                        x_best = x_restart.copy()
                        f_best = f_restart
                    # Reset temperature for new phase
                    T_initial = max(T_min * 10, f_best * 0.1 + 1.0)
                    step_scale = 1.0
                    stagnation_count = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
