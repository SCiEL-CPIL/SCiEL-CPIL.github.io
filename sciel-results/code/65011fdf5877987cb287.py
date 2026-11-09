"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based evolutionary optimizer with adaptive step size for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 330s limit, leave 50s margin for safety

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

        # Phase 1: Structured + random sampling for early progress
        points = []
        points.append(np.zeros(d))
        for corner in [(1,1), (-1,1), (1,-1), (-1,-1)]:
            pts = []
            for i in range(d):
                if i % 2 == 0:
                    pts.append(corner[0] * 3.0)
                else:
                    pts.append(corner[1] * 3.0)
            points.append(np.array(pts))
        
        for i in range(d):
            x = np.zeros(d)
            x[i] = 2.5
            points.append(x.copy())
            x[i] = -2.5
            points.append(x.copy())

        n_random = max(1, min(15, budget - len(points)))
        for _ in range(n_random):
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

        # Phase 2: (1+lambda)-ES with adaptive step
        if calls_made < budget and time.time() < deadline:
            sigma = 0.5
            x = best_x.copy()
            fx = best_fx
            lambda_ = 4
            success_count = 0
            total_count = 0
            recent_successes = []

            while time_ok():
                candidates = []
                for _ in range(lambda_):
                    y = x + sigma * rng.standard_normal(d)
                    y = np.clip(y, -5, 5)
                    candidates.append(y)

                cands_fx = []
                for y in candidates:
                    if not time_ok():
                        break
                    fy = call_func(y)
                    cands_fx.append(fy)

                if not time_ok():
                    break

                best_idx = np.argmin(cands_fx)
                best_cand_fx = cands_fx[best_idx]
                best_cand = candidates[best_idx]

                total_count += len(cands_fx)
                improved = best_cand_fx < fx

                if improved:
                    x = best_cand
                    fx = best_cand_fx
                    success_count += 1
                    sigma *= 1.15
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.88
                    sigma = max(sigma, 0.001)

                recent_successes.append(1 if improved else 0)
                if len(recent_successes) > 20:
                    recent_successes.pop(0)

                if len(recent_successes) >= 10:
                    rate = sum(recent_successes) / len(recent_successes)
                    if rate > 0.6:
                        sigma *= 1.05
                    elif rate < 0.2:
                        sigma *= 0.95

        # Phase 3: Final polish with very small steps
        if calls_made < budget and time.time() < deadline:
            polish_calls = min(50, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.01 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
