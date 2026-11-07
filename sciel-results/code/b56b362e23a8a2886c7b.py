"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive sigma and time guard.
    Designed for BBOB f4 (Büche-Rastrigin) with 500 evaluations in dim=5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        # State
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Coarse global search with multiple random starts ---
        # Use a fraction of budget for global exploration
        n_starts = 10
        per_start_budget = max(5, (budget * 2) // 3 // n_starts)  # ~66 per start

        rng = np.random.default_rng(42)

        for s in range(n_starts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Random start in the box
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)

            sigma = 1.5
            stagnation = 0
            max_stagnation = 15

            for i in range(per_start_budget):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate trial
                direction = rng.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.15
                    stagnation = 0
                else:
                    sigma *= 0.85
                    stagnation += 1
                    if stagnation >= max_stagnation:
                        # Try a larger jump to escape local minimum
                        if call_count < budget and time.time() < deadline:
                            jump = rng.uniform(-2, 2, dim)
                            y = np.clip(x + jump, -5, 5)
                            fy = evaluate(y)
                            if fy < fx:
                                x = y
                                fx = fy
                                sigma = 1.0
                                stagnation = 0
                            else:
                                sigma *= 0.7
                        break

            # After each start, if we found something better, continue with refinement
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: Fine-grained local refinement around best point ---
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            stagnation = 0
            max_stagnation = 20

            while call_count < budget and time.time() < deadline:
                direction = rng.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                    stagnation = 0
                else:
                    sigma *= 0.88
                    stagnation += 1
                    if stagnation >= max_stagnation:
                        # Adaptive restart: try a moderate perturbation
                        if call_count < budget and time.time() < deadline:
                            jump = rng.normal(0, 0.5, dim)
                            y = np.clip(x + jump, -5, 5)
                            fy = evaluate(y)
                            if fy < fx:
                                x = y
                                fx = fy
                                sigma = 0.3
                                stagnation = 0
                            else:
                                sigma *= 0.7
                        else:
                            break

        # --- Phase 3: Very fine local search if budget remains ---
        if best_x is not None and call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            stagnation = 0
            max_stagnation = 15

            while call_count < budget and time.time() < deadline:
                direction = rng.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05
                    stagnation = 0
                else:
                    sigma *= 0.92
                    stagnation += 1
                    if stagnation >= max_stagnation:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
