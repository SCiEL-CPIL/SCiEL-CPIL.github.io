"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, axis probing, and restarts for bent cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_time():
            return time.time() < deadline and calls_used < budget

        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 0: Probe each axis from center to identify the hard direction
        # Bent cigar: f(x) = 100*(x1^2 + sum_{i=2}^n (x_i/100)^2)
        # After rotation/shift, one direction has large coefficient
        center = np.zeros(dim)
        fc = evaluate(center)

        # Probe along each axis with a moderate step to find the steepest direction
        probe_step = 1.0
        axis_costs = []
        for i in range(dim):
            x_plus = center.copy()
            x_plus[i] += probe_step
            fp = evaluate(x_plus)
            # Cost change from center in this direction
            axis_costs.append(abs(fp - fc))

        # The hard direction has the largest cost change
        hard_idx = int(np.argmax(axis_costs))

        # Phase 1: (1+1)-ES with adaptive sigma, biased toward exploiting easy directions
        # Start from center (often close to optimum for shifted/rotated bent cigar)
        x = center.copy()
        fx = fc

        sigma = 1.5
        no_improve = 0
        max_no_improve = 40

        # For the first phase, use a single (1+1)-ES run with aggressive adaptation
        while check_time():
            y = x + sigma * rng.standard_normal(dim)
            fy = evaluate(y)

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma *= 1.25  # increase step on success
            else:
                no_improve += 1
                sigma *= 0.82  # decrease step on failure

            if no_improve >= max_no_improve:
                break

        # Phase 2: Multiple short restarts with decreasing sigma to refine
        for restart in range(15):
            if not check_time():
                break

            # Initialize near best found
            if best_x is not None:
                x = best_x + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = evaluate(x)

            sigma = 0.8 * (0.7 ** restart)  # decreasing initial sigma
            no_improve = 0
            max_no_improve = 25 + 5 * restart

            while check_time() and no_improve < max_no_improve:
                y = x + sigma * rng.standard_normal(dim)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.2
                else:
                    no_improve += 1
                    sigma *= 0.8

            if calls_used >= budget:
                break

        # Phase 3: Final high-precision polish with very small steps
        if best_x is not None and check_time():
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            no_improve = 0

            for _ in range(min(100, budget - calls_used)):
                if not check_time():
                    break
                y = x + sigma * rng.standard_normal(dim)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.1
                else:
                    no_improve += 1
                    sigma *= 0.7
                if no_improve >= 20:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
