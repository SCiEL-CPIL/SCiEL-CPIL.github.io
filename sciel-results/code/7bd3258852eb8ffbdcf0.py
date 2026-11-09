"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Lattice global search followed by adaptive (1+1)-ES refinement with final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls = 0

        def evaluate(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return x, fx

        # Phase 1: Deterministic lattice search on a 3^5 grid
        # Covers [-5, 0, 5] in each dimension = 243 evaluations
        levels = np.linspace(-5.0, 5.0, 3)  # [-5, 0, 5]
        best_x = None
        best_f = np.inf

        for combo in itertools.product(*[levels] * n):
            if time.time() > deadline or calls >= budget:
                break
            x = np.array(combo, dtype=float)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = float(func(best_x))
            calls += 1

        # Phase 2: Adaptive (1+1)-ES refinement
        # Start with a step size of 2.0 (half the grid spacing)
        step_size = 2.0
        no_improve = 0
        max_no_improve = 15
        rng = np.random.default_rng(42)

        while calls < budget and time.time() < deadline:
            # Generate a random direction
            direction = rng.normal(size=n)
            direction /= (np.linalg.norm(direction) + 1e-15)

            x_try = best_x + direction * step_size
            x_try = np.clip(x_try, -5.0, 5.0)

            fx_try = float(func(x_try))
            calls += 1

            if fx_try < best_f:
                best_x = x_try.copy()
                best_f = fx_try
                no_improve = 0
                step_size = min(step_size * 1.05, 10.0)
            else:
                no_improve += 1
                step_size = max(step_size * 0.85, 1e-6)

                if no_improve >= max_no_improve:
                    # Try a coordinate-wise refinement pass
                    improved_coord = False
                    for d in range(n):
                        if calls >= budget or time.time() > deadline:
                            break
                        for sign in [1.0, -1.0]:
                            if calls >= budget or time.time() > deadline:
                                break
                            x_c = best_x.copy()
                            x_c[d] += sign * step_size
                            x_c = np.clip(x_c, -5.0, 5.0)
                            fx_c = float(func(x_c))
                            calls += 1
                            if fx_c < best_f:
                                best_x = x_c.copy()
                                best_f = fx_c
                                improved_coord = True
                                no_improve = 0
                                step_size = max(step_size * 1.1, 0.01)
                    if improved_coord:
                        no_improve = 0
                    else:
                        # Reset step size and try again
                        step_size = 0.5
                        no_improve = 0

        # Phase 3: Final high-precision polish pass
        # Use remaining budget for very fine local search
        polish_budget = max(0, budget - calls)
        if polish_budget > 10 and time.time() < deadline:
            step_size = 0.01
            for _ in range(min(polish_budget // 2, 100)):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(n):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1.0, -1.0]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_p = best_x.copy()
                        x_p[d] += sign * step_size
                        x_p = np.clip(x_p, -5.0, 5.0)
                        fx_p = float(func(x_p))
                        calls += 1
                        if fx_p < best_f:
                            best_x = x_p.copy()
                            best_f = fx_p
                            improved = True
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
