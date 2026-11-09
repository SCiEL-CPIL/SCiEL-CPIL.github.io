"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with multi-restart and final polish for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 330s limit, leave 60s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Compact initial sampling for early progress
        points = [np.zeros(d)]
        # Axis-aligned points at moderate distance
        for i in range(d):
            for sign in (1, -1):
                x = np.zeros(d)
                x[i] = sign * 2.0
                points.append(x)
        # A few random points in the box
        for _ in range(8):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: (1+lambda)-ES with adaptive sigma and restarts
        lambda_ = 4
        sigma = 1.0
        x = best_x.copy()
        fx = best_fx
        no_improve = 0
        max_no_improve = 40  # restart after 40 iterations without improvement

        while time_ok():
            # Generate lambda candidates
            candidates = []
            for _ in range(lambda_):
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                candidates.append(y)

            # Evaluate all candidates
            cands_fx = []
            for y in candidates:
                if not time_ok():
                    break
                fy = call_func(y)
                cands_fx.append(fy)

            if len(cands_fx) < lambda_:
                break

            # Find best candidate
            best_idx = np.argmin(cands_fx)
            best_cand_fx = cands_fx[best_idx]
            best_cand = candidates[best_idx]

            if best_cand_fx < fx:
                x = best_cand
                fx = best_cand_fx
                no_improve = 0
                sigma *= 1.3  # expand on success
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                no_improve += 1
                sigma *= 0.9  # shrink on failure
                sigma = max(sigma, 0.005)

            # Restart if stuck: jump to a new random point
            if no_improve >= max_no_improve and calls_made < budget - 20:
                no_improve = 0
                sigma = 1.5
                x = rng.uniform(-4, 4, d)
                x = np.clip(x, -5, 5)
                fx = call_func(x)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        # Phase 3: Final high-precision polish around best point
        if calls_made < budget - 5 and time.time() < deadline:
            sigma = 0.1
            x = best_x.copy()
            fx = best_fx
            polish_lambda = 3

            for _ in range(30):
                if not time_ok():
                    break
                candidates = []
                for _ in range(polish_lambda):
                    y = x + sigma * rng.standard_normal(d)
                    y = np.clip(y, -5, 5)
                    candidates.append(y)

                cands_fx = []
                for y in candidates:
                    if not time_ok():
                        break
                    fy = call_func(y)
                    cands_fx.append(fy)

                if len(cands_fx) < polish_lambda:
                    break

                best_idx = np.argmin(cands_fx)
                best_cand_fx = cands_fx[best_idx]

                if best_cand_fx < fx:
                    x = candidates[best_idx]
                    fx = best_cand_fx
                    sigma *= 1.1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
                    sigma = max(sigma, 0.001)

        return best_x, best_fx
# EVOLVE-BLOCK-END
