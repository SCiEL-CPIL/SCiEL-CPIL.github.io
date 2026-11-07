# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start portfolio optimizer for Katsuura (f23) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _local_search(self, func, x0, fx0, budget, sigma0, rng, deadline, use_15_rule=False):
        """(1+1)-ES local refinement with adaptive sigma."""
        x = x0.copy()
        fx = fx0
        sigma = sigma0
        calls = 0
        success_count = 0
        window = 10
        
        for _ in range(budget):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x = y
                fx = fy
                success_count += 1
            else:
                pass
            
            if use_15_rule:
                if success_count >= window * 0.5:
                    sigma *= 1.25
                    success_count = 0
                    window = max(10, int(window * 1.3))
                elif success_count == 0:
                    sigma *= 0.8
                    success_count = 0
                    window = max(10, int(window * 1.3))
            else:
                if fy < fx:
                    sigma *= 1.2
                else:
                    sigma *= 0.9
            
            if sigma < 1e-7:
                break
        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Exploration with diverse starts
        n_starts = min(12, max(8, budget // 40))
        per_start_explore = max(20, (budget // 3) // n_starts)

        # Generate diverse starting points
        start_points = []
        # Key structural points for Katsuura
        structural = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.full(dim, 0.5),
            np.full(dim, -0.5),
            np.array([1, -1, 1, -1, 1][:dim]),
            np.array([-1, 1, -1, 1, -1][:dim]),
            np.array([2, -2, 2, -2, 2][:dim]),
            np.array([-2, 2, -2, 2, -2][:dim]),
        ]
        
        for s in structural[:n_starts]:
            start_points.append(np.clip(s + rng.uniform(-0.3, 0.3, dim), -5, 5))
        
        while len(start_points) < n_starts:
            start_points.append(rng.uniform(-3, 3, dim))

        # Shuffle for diversity
        idx = rng.permutation(n_starts)
        start_points = [start_points[i] for i in idx]

        for i in range(n_starts):
            if time.time() > deadline:
                break
            x0 = np.clip(start_points[i], -5, 5)
            fx0 = func(x0)
            total_calls += 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            remaining = per_start_explore
            if total_calls + remaining > budget:
                remaining = max(0, budget - total_calls)
            
            if remaining > 0:
                x, fx, calls = self._local_search(func, x0, fx0, remaining, 2.0, rng, deadline, use_15_rule=False)
                total_calls += calls
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 2: Exploitation on best candidate
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            # Use finer sigma for exploitation
            x, fx, calls = self._local_search(func, best_x, best_f, remaining, 0.3, rng, deadline, use_15_rule=True)
            total_calls += calls
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 3: Final fine-tuning if budget remains
        if best_x is not None and total_calls < budget and time.time() < deadline - 10:
            remaining = budget - total_calls
            x, fx, calls = self._local_search(func, best_x, best_f, remaining, 0.1, rng, deadline, use_15_rule=True)
            total_calls += calls
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
