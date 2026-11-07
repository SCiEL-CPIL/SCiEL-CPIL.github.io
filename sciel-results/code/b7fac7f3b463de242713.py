"""
Portfolio optimizer for BBOB f3 (Rastrigin) with integer-grid initialization,
adaptive (1+1)-ES local search, and Nelder-Mead final polish.
Strategy:
1. Structured initialization exploiting Rastrigin's periodic structure (local minima at integers).
2. Adaptive (1+1)-ES with success-rate-based sigma adaptation for local refinement.
3. Nelder-Mead for final high-precision polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls_made = 0
        max_calls = self.budget
        dim = self.dim

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= max_calls

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Structured initialization exploiting Rastrigin's periodic structure
        # Rastrigin has local minima at integer points. The hidden shift means the
        # global min is at some shifted integer grid point. Sample near integers.
        n_explore = min(40, max_calls // 10)
        
        # Generate candidate points near integer grid
        # Sample each dimension from {-4, -3, -2, -1, 0, 1, 2, 3, 4} with small jitter
        candidates = [np.zeros(dim)]
        
        # Add structured points: sample from integer grid with jitter
        # Use a space-filling approach: sample random integer vectors
        for _ in range(n_explore):
            # Random integer point in [-4, 4]^dim
            int_pt = rng.integers(-4, 5, dim).astype(float)
            # Add small jitter to avoid exactly hitting (in case shift is non-integer)
            jitter = rng.uniform(-0.3, 0.3, dim)
            pt = int_pt + jitter
            candidates.append(np.clip(pt, -5, 5))
        
        # Also add some purely random points for coverage
        for _ in range(5):
            candidates.append(rng.uniform(-4.5, 4.5, dim))

        for x in candidates:
            if check_deadline():
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            update_best(x_clipped, f)

        if best_x is None:
            x = np.zeros(dim)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 2: Adaptive (1+1)-ES local search with success-rate sigma adaptation
        remaining = max_calls - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Reserve budget for Nelder-Mead polish
        nm_budget = min(80, remaining // 5)
        es_budget = remaining - nm_budget

        if es_budget > 10:
            x = best_x.copy()
            f = best_f
            sigma = 1.0
            min_sigma = 1e-6
            max_sigma = 5.0
            
            # Success rate window for sigma adaptation
            window = []
            window_size = 15
            
            for _ in range(es_budget):
                if check_deadline():
                    break

                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < f:
                    x = y
                    f = fy
                    window.append(1)
                    # Update global best
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    window.append(0)

                # Adapt sigma based on success rate in recent window
                if len(window) >= window_size:
                    recent = window[-window_size:]
                    success_rate = sum(recent) / window_size
                    if success_rate > 0.35:
                        sigma = min(sigma * 1.15, max_sigma)
                    elif success_rate < 0.15:
                        sigma = max(sigma * 0.75, min_sigma)
                    window = window[-window_size:]

        # Phase 3: Nelder-Mead final high-precision polish
        if not check_deadline() and calls_made < max_calls:
            nm_calls = max_calls - calls_made
            if nm_calls > 10:
                try:
                    from scipy.optimize import minimize

                    nm_calls_remaining = max_calls - calls_made

                    def wrapper(x):
                        nonlocal calls_made
                        if calls_made >= max_calls:
                            return best_f
                        calls_made += 1
                        return func(np.clip(x, -5, 5))

                    # Run Nelder-Mead from current best with tight tolerances
                    result = minimize(
                        wrapper,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': nm_calls_remaining,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )

                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
