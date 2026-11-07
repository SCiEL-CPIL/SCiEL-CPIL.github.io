"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: multi-start (1+1)-ES exploration + Rastrigin-aware refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start (1+1)-ES with dynamic budget allocation
        # Each restart gets a budget; if it stagnates (no improvement in ~20 calls),
        # we reset sigma and continue, or move to next restart if budget exhausted.
        n_restarts = 6
        per_restart = 70  # ~420 calls for exploration

        for r in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Initialize this restart
            if r == 0:
                x = np.zeros(dim)
            else:
                # Perturb from global best or random
                if rng.random() < 0.3:
                    x = best_x + 2.0 * rng.standard_normal(dim) if best_x is not None else rng.uniform(-4, 4, dim)
                else:
                    x = rng.uniform(-4, 4, dim)
            x = np.clip(x, -5, 5)
            fx = eval_func(x)

            sigma = 1.5
            # Success rate tracking for adaptive sigma
            window = 10
            recent = [False] * window
            idx = 0
            last_improve_call = calls

            for step in range(per_restart):
                if calls >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                    last_improve_call = calls
                else:
                    recent[idx % window] = False
                idx += 1

                # Adapt sigma based on success rate
                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma *= 1.15
                    elif sr < 0.2:
                        sigma *= 0.85
                    # else keep same

                sigma = np.clip(sigma, 0.05, 3.0)

                # Stagnation check: if no improvement in 25 calls, reset sigma and break
                if calls - last_improve_call > 25:
                    # Reset sigma to a moderate value and try a few more steps
                    sigma = 1.0
                    last_improve_call = calls
                    # If still stuck after reset, break to next restart
                    if calls - last_improve_call > 25:
                        break

        # Phase 2: Rastrigin-aware refinement
        if best_x is not None and calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f

            # Strategy 1: Coordinate-wise refinement with decreasing step
            step = 0.3
            for _ in range(40):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_local.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break

            # Strategy 2: Try snapping to nearby "integer-like" points
            # Rastrigin has minima at integer coordinates (after rotation/shift)
            if calls < budget and time.time() < deadline:
                candidates = []
                for d in range(dim):
                    base = x_local.copy()
                    for offset in [-1, 0, 1]:
                        y = base.copy()
                        y[d] = round(y[d]) + offset
                        y = np.clip(y, -5, 5)
                        candidates.append(y)
                candidates.append(np.clip(np.round(x_local), -5, 5))

                seen = set()
                for y in candidates:
                    if calls >= budget or time.time() > deadline:
                        break
                    key = tuple(np.round(y, 6))
                    if key not in seen:
                        seen.add(key)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy

            # Strategy 3: Final fine-tuning with very small steps
            if calls < budget and time.time() < deadline:
                if best_f < f_local:
                    x_local = best_x.copy()
                    f_local = best_f
                step = 0.05
                for _ in range(30):
                    if calls >= budget or time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        for sign in [1, -1]:
                            if calls >= budget or time.time() > deadline:
                                break
                            y = x_local.copy()
                            y[d] += sign * step
                            y = np.clip(y, -5, 5)
                            fy = eval_func(y)
                            if fy < f_local:
                                x_local, f_local = y, fy
                                improved = True
                                break
                    if not improved:
                        step *= 0.5
                        if step < 0.005:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
