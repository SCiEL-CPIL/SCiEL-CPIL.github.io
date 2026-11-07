import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with scheduled restarts and local polish for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 0: quick multi-start to seed a rough best (helps AOCC early progress)
        for _ in range(min(12, budget)):
            if not time_ok() or calls >= budget:
                break
            x0 = rng.uniform(-5.0, 5.0, d)
            fx0 = float(func(x0))
            calls += 1
            record(x0, fx0)

        # Reserve budget for final polish
        polish_budget = max(40, int(0.12 * budget))
        es_limit = budget - polish_budget

        # Phase 1: adaptive (1+1)-ES with scheduled restarts (exploration -> exploitation)
        # Each entry: (initial_sigma, start_mode)  mode 0=at best, 1=near best, 2=random
        schedule = [
            (1.5, 1),
            (1.2, 1),
            (0.8, 1),
            (0.5, 0),
            (0.3, 0),
            (0.2, 0),
            (0.12, 0),
            (0.06, 0),
            (0.03, 0),
            (0.015, 0),
            (0.008, 0),
            (0.004, 0),
        ]

        c_sigma = 0.5
        zeta = 0.33
        for sigma0, mode in schedule:
            if not time_ok() or calls >= es_limit:
                break
            if best_x is None:
                x = rng.uniform(-5.0, 5.0, d)
            elif mode == 2:
                x = rng.uniform(-5.0, 5.0, d)
            elif mode == 1:
                x = np.clip(best_x + 0.6 * sigma0 * rng.standard_normal(d), -5.0, 5.0)
            else:
                x = best_x.copy()

            fx = float(func(x))
            calls += 1
            record(x, fx)
            sigma = sigma0

            ew_succ = 0.0
            ew_total = 0.0
            no_improve = 0
            stag_limit = 45

            while time_ok() and calls < es_limit:
                z = rng.standard_normal(d)
                y = np.clip(x + sigma * z, -5.0, 5.0)
                fy = float(func(y))
                calls += 1
                success = fy < fx
                ew_succ = 0.8 * ew_succ + 0.2 * (1.0 if success else 0.0)
                ew_total = 0.8 * ew_total + 0.2
                sr = ew_succ / max(ew_total, 1e-12)
                if success:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                sigma *= np.exp(c_sigma * (sr - zeta))
                sigma = float(np.clip(sigma, 1e-7, 5.0))
                if no_improve > stag_limit:
                    break

        # Phase 2: local polish (coordinate descent + adaptive random fine steps)
        if best_x is not None and time_ok() and calls < budget:
            x = best_x.copy()
            fx = best_fx
            # coordinate descent with adaptive step
            for _ in range(3):
                if not time_ok() or calls >= budget:
                    break
                for coord in range(d):
                    if not time_ok() or calls >= budget:
                        break
                    for direction in (1, -1):
                        if not time_ok() or calls >= budget:
                            break
                        step = 0.3
                        for _ in range(10):
                            if not time_ok() or calls >= budget:
                                break
                            y = x.copy()
                            y[coord] = np.clip(y[coord] + direction * step, -5.0, 5.0)
                            fy = float(func(y))
                            calls += 1
                            if fy < fx:
                                x = y
                                fx = fy
                                if fx < best_fx:
                                    best_fx = fx
                                    best_x = x.copy()
                                step *= 0.7
                            else:
                                break
            # adaptive random fine steps
            sigma = 0.05
            while time_ok() and calls < budget:
                y = np.clip(x + sigma * rng.standard_normal(d), -5.0, 5.0)
                fy = float(func(y))
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85

        # Safety: ensure a valid result
        if best_x is None:
            x = np.zeros(d)
            best_fx = float(func(x))
            best_x = x.copy()

        return best_x, best_fx
