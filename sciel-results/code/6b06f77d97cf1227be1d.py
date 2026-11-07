import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: determine gradient, jump to exact corner, then deterministic polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295.0

        def time_ok():
            return time.time() < deadline

        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1

        # Phase 1: Determine gradient signs for first dim-1 dimensions
        # Use a large step (h=4) to robustly detect direction despite hidden shift
        h = 4.0
        x0 = np.zeros(dim)
        f0 = fx_best
        grad_signs = np.zeros(dim)
        
        # We probe dim-1 dimensions. We leave the last one for the candidate evaluation.
        # This allows us to evaluate 2 candidate vertices in the last 2 calls.
        probe_dims = dim - 1
        
        for i in range(probe_dims):
            if not time_ok() or calls >= budget:
                break
            e = np.zeros(dim)
            e[i] = h
            xp = np.clip(x0 + e, -5, 5)
            fp = func(xp)
            calls += 1
            if fp < f0:
                grad_signs[i] = -1.0  # decreasing in +e_i direction, min at +5
            else:
                grad_signs[i] = 1.0   # increasing in +e_i direction, min at -5

        # Phase 2: Construct two candidate vertices for the last dimension
        # The last dimension index is dim-1
        last_dim = dim - 1
        x_candidate = np.zeros(dim)
        for i in range(probe_dims):
            x_candidate[i] = 5.0 if grad_signs[i] == -1.0 else -5.0
        
        # Candidate A: last dim = 5
        x_candidate[last_dim] = 5.0
        fx_cand_a = func(x_candidate)
        calls += 1
        if fx_cand_a < fx_best:
            x_best = x_candidate.copy()
            fx_best = fx_cand_a

        # Candidate B: last dim = -5
        x_candidate[last_dim] = -5.0
        fx_cand_b = func(x_candidate)
        calls += 1
        if fx_cand_b < fx_best:
            x_best = x_candidate.copy()
            fx_best = fx_cand_b

        # Phase 3: Deterministic high-precision local search around the corner
        # Since the function is linear, we're at the global minimum.
        # Use a grid-based search with decreasing step sizes to handle any
        # numerical edge cases or slight non-linearity from the hidden transform.
        
        if calls < budget and time_ok():
            # Multi-scale deterministic search
            scales = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            for scale in scales:
                if not time_ok() or calls >= budget:
                    break
                for i in range(dim):
                    if not time_ok() or calls >= budget:
                        break
                    for direction in [1.0, -1.0]:
                        if not time_ok() or calls >= budget:
                            break
                        y = x_best.copy()
                        y[i] += direction * scale
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < fx_best:
                            x_best = y.copy()
                            fx_best = fy

        # Phase 4: Final fine-tuning with very small steps if budget remains
        if calls < budget and time_ok():
            rng = np.random.default_rng(123)
            sigma = 1e-4
            for _ in range(min(50, budget - calls)):
                if not time_ok() or calls >= budget:
                    break
                y = np.clip(x_best + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx_best:
                    x_best = y.copy()
                    fx_best = fy

        return x_best, fx_best
