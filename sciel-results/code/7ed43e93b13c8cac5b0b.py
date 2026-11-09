"""Optimizer for BBOB-style f19 (Griewank-Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart adaptive optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: broad initial sampling to find a good starting region
        # Use a grid-like spread plus random points
        n_initial = min(30, budget // 4)
        for i in range(n_initial):
            if time.time() > deadline:
                break
            if i < 10:
                # structured points: corners and mid-points of sub-boxes
                # 3^5 = 243 possible grid points on {-4,-2,0,2,4}
                # pick a deterministic subset
                idx = i
                x = np.array([
                    -4 + 2 * ((idx >> (4 * d)) & 1) + 2 * ((idx >> (3 * d)) & 1)
                    for d in range(dim)
                ], dtype=float)
                # Use a simpler structured approach: Halton-like
                x = np.array([
                    ((i * (d + 1) * 7919) % 100) / 100.0 * 8.0 - 4.0
                    for d in range(dim)
                ])
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            try_call(x)

        # Phase 2: local refinement from the best point using adaptive (1+1)-ES
        # with restarts if no progress
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 1.5
            no_improve = 0
            max_no_improve = 40

            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

                if no_improve >= max_no_improve:
                    # Restart from a new random point or slightly perturbed best
                    if time.time() > deadline:
                        break
                    if rng.random() < 0.5:
                        x = rng.uniform(-4.5, 4.5, dim)
                        fx = try_call(x)
                        if fx is None:
                            break
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        x = np.clip(best_x + rng.normal(0, 2.0, dim), -5, 5)
                        fx = try_call(x)
                        if fx is None:
                            break
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    sigma = 1.0
                    no_improve = 0

        # Phase 3: if we still have budget, do a few more random explorations
        # to potentially find a better basin
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            n_explore = min(remaining // 3, 20)
            for _ in range(n_explore):
                if time.time() > deadline or calls >= budget:
                    break
                x = rng.uniform(-4.8, 4.8, dim)
                fx = try_call(x)
                if fx is None:
                    break
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    # Quick local refinement from this new best
                    x_local = best_x.copy()
                    fx_local = best_f
                    sigma_local = 0.8
                    for _ in range(min(20, budget - calls)):
                        if time.time() > deadline or calls >= budget:
                            break
                        y = np.clip(x_local + sigma_local * rng.standard_normal(dim), -5, 5)
                        fy = try_call(y)
                        if fy is None:
                            break
                        if fy < fx_local:
                            x_local, fx_local = y, fy
                            sigma_local *= 1.2
                        else:
                            sigma_local *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
