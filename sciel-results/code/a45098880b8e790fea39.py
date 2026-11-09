import numpy as np
import time

class Optimizer:
    """Portfolio optimizer: adaptive exploration + CMA-ES style search with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin

        # Counters
        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # ---- Phase 1: Diverse initial exploration ----
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # More diverse initial starts covering the search space
        starts = [
            np.array([0.0] * dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            np.array([4.0] * dim),
            np.array([-4.0] * dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for s in starts:
            if calls[0] >= budget:
                break
            x = np.clip(s, -5, 5)
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 2: Adaptive multi-start local search ----
        # Use remaining budget for adaptive local search from best point
        center = best_x.copy()
        cf = best_f
        sigma = 1.0
        lam = 6  # population size

        # Track improvement history for adaptive step size
        recent_improvements = []

        while calls[0] < budget and time.time() < deadline:
            # Generate population with adaptive sigma
            improved_any = False
            for i in range(lam):
                if calls[0] >= budget or time.time() > deadline:
                    break
                child = np.clip(center + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    cf_c = safe_func(child)
                except StopIteration:
                    break
                if cf_c < cf:
                    cf = cf_c
                    center = child
                    improved_any = True
                    recent_improvements.append(1)
                else:
                    recent_improvements.append(0)

            # Keep only last 10 improvement records
            if len(recent_improvements) > 10:
                recent_improvements.pop(0)

            # Adapt sigma based on recent improvement rate
            if recent_improvements:
                imp_rate = sum(recent_improvements) / len(recent_improvements)
                if imp_rate > 0.5:
                    sigma *= 1.15  # increasing step when improving
                elif imp_rate < 0.2:
                    sigma *= 0.75  # decreasing step when stuck
                else:
                    sigma *= 0.95  # moderate decrease

            if sigma < 1e-7:
                break

        # ---- Phase 3: High-precision polish pass ----
        # Use remaining budget for fine-grained local refinement
        polish_calls_remaining = budget - calls[0]
        if polish_calls_remaining > 10 and time.time() < deadline:
            center = best_x.copy()
            cf = best_f
            sigma = 0.1  # start with smaller step for polish
            
            # Coordinate-wise refinement with adaptive step
            for _ in range(min(50, polish_calls_remaining // 2)):
                if calls[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if time.time() > deadline:
                        break
                    for sign in (1, -1):
                        if calls[0] >= budget:
                            break
                        x2 = center.copy()
                        x2[d] = np.clip(center[d] + sign * sigma, -5, 5)
                        try:
                            f2 = safe_func(x2)
                        except StopIteration:
                            break
                        if f2 < cf:
                            cf = f2
                            center = x2
                            improved = True
                    if time.time() > deadline:
                        break
                if improved:
                    sigma *= 0.85
                else:
                    sigma *= 0.6
                if sigma < 1e-8:
                    break
            
            best_x = center
            best_f = cf

        return best_x, best_f
