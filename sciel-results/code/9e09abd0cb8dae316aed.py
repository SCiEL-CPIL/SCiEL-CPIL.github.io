"""Adaptive (1+1)-ES with multi-start for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Adaptive (1+1)-ES with multi-start for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Quick diverse starts to find a good basin early
        starts = [np.zeros(dim)]
        for scale in [1.0, 2.0, 3.0, 4.0]:
            starts.append(np.full(dim, scale))
            starts.append(np.full(dim, -scale))
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        for s in starts:
            if n_calls[0] >= budget or time.time() > deadline:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        # Phase 2: Adaptive (1+1)-ES with restarts
        def run_es(start_x, max_calls):
            nonlocal best_x, best_f
            n = dim
            x = start_x.copy()
            sigma = 1.0
            local_best_x = x.copy()
            local_best_f = safe_func(x)
            
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            no_improve = 0
            stall_threshold = 20
            window = []
            window_size = 10

            while n_calls[0] < max_calls and time.time() < deadline and no_improve < stall_threshold:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)

                if fy < local_best_f:
                    x = y.copy()
                    local_best_f = fy
                    local_best_x = x.copy()
                    no_improve = 0
                    window.append(1)
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                else:
                    no_improve += 1
                    window.append(0)

                # Adapt sigma based on success rate
                if len(window) >= window_size:
                    sr = sum(window) / len(window)
                    if sr > 0.35:
                        sigma *= 1.25
                    elif sr < 0.15:
                        sigma *= 0.8
                    window = window[-window_size // 2:]

                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 2.0)

            return local_best_x, local_best_f

        # Run ES from the best start
        es_x, es_f = run_es(best_x, int(budget * 0.7))
        if es_f < best_f:
            best_f = es_f
            best_x = es_x.copy()

        # Phase 3: Polish with smaller step size
        if n_calls[0] < budget and time.time() < deadline:
            polish_sigma = 0.05
            polish_window = []
            polish_window_size = 6

            while n_calls[0] < budget and time.time() < deadline:
                candidate = best_x + polish_sigma * rng.standard_normal(dim)
                candidate = np.clip(candidate, -5.0, 5.0)
                fx = safe_func(candidate)

                if fx < best_f:
                    best_x = candidate.copy()
                    best_f = fx
                    polish_window.append(1)
                    polish_sigma *= 0.8
                else:
                    polish_window.append(0)
                    polish_sigma *= 0.95

                if len(polish_window) >= polish_window_size:
                    sr = sum(polish_window) / len(polish_window)
                    if sr > 0.4:
                        polish_sigma *= 1.3
                    elif sr < 0.1:
                        polish_sigma *= 0.6
                    polish_window = polish_window[-polish_window_size // 2:]

                polish_sigma = max(polish_sigma, 1e-10)
                polish_sigma = min(polish_sigma, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
