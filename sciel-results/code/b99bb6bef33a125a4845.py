"""Optimizer for BBOB-style Katsuura f23 on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """Hybrid global-local optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin under 300s
        n = self.dim
        rng = np.random.default_rng(12345)

        state = {"calls": 0, "x_best": None, "f_best": np.inf}

        def time_ok():
            return time.time() < deadline and state["calls"] < self.budget

        def evaluate(x):
            fx = float(func(x))
            state["calls"] += 1
            if fx < state["f_best"]:
                state["f_best"] = fx
                state["x_best"] = x.copy()
            return fx

        # Phase 1: Global random sampling to find a good basin
        # Katsuura f23 has many local minima; global sampling helps locate the basin
        n_global = max(30, self.budget // 8)  # ~60 calls
        for _ in range(n_global):
            if not time_ok():
                break
            r = rng.random()
            if r < 0.4:
                x = rng.uniform(-5, 5, n)
            elif r < 0.7:
                x = rng.uniform(-3, 3, n)
            else:
                x = rng.uniform(-1.5, 1.5, n)
            evaluate(x)

        # Phase 2: Local refinement from the best point
        if state["x_best"] is not None and time_ok():
            state["x_best"], state["f_best"] = self._local_search(
                func, state["x_best"], state["f_best"], rng, time_ok, state
            )

        # Phase 3: Additional restarts if budget remains
        while time_ok():
            # Check if we have enough budget for a meaningful restart
            remaining = self.budget - state["calls"]
            if remaining < 40:
                break
            # Global sample
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx < state["f_best"] - 1e-10:
                # New best found; do local search
                if time_ok():
                    state["x_best"], state["f_best"] = self._local_search(
                        func, state["x_best"], state["f_best"], rng, time_ok, state
                    )
            else:
                # No improvement; continue global sampling
                continue

        # Ensure valid return
        if state["x_best"] is None:
            x = rng.uniform(-5, 5, n)
            state["x_best"] = x
            state["f_best"] = float(func(x))
            state["calls"] += 1

        return state["x_best"], state["f_best"]

    def _local_search(self, func, x, fx, rng, time_ok, state):
        """(mu+lambda)-ES local search with adaptive sigma."""
        n = self.dim
        x_best = x.copy()
        f_best = fx
        sigma = 0.5
        lam = 6
        mu = 3

        success_window = []
        max_window = 10

        while time_ok():
            # Generate lambda candidates
            candidates = []
            for _ in range(lam):
                step = sigma * rng.standard_normal(n)
                y = np.clip(x_best + step, -5, 5)
                candidates.append(y)

            # Evaluate candidates, track best
            best_y = x_best.copy()
            best_f = f_best
            for y in candidates:
                if not time_ok():
                    break
                fy = float(func(y))
                state["calls"] += 1
                if fy < best_f:
                    best_f = fy
                    best_y = y.copy()

            if best_f < f_best:
                x_best = best_y.copy()
                f_best = best_f
                success_window.append(1)
            else:
                success_window.append(0)

            if len(success_window) > max_window:
                success_window.pop(0)

            # Adapt sigma
            rate = sum(success_window) / len(success_window)
            if rate > 0.4:
                sigma *= 1.15
            elif rate < 0.15:
                sigma *= 0.75
            else:
                sigma *= 0.95

            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 5.0)

            # Stop if converged
            if sigma < 1e-7:
                break

        return x_best, f_best
