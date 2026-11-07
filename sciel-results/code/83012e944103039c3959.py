import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit

        def time_left():
            return deadline - time.time()

        calls = 0
        all_points = []
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64).copy()
            fx = float(func(x))
            calls += 1
            all_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nm_minimize(start_x, max_calls=60):
            """Nelder-Mead with explicit call budget tracking."""
            nonlocal calls, best_x, best_f
            if calls >= budget - 5 or time_left() < 2.0:
                return
            try:
                from scipy.optimize import minimize
                nm_calls = [0]
                def wrapped_func(x):
                    nonlocal calls, best_x, best_f
                    if nm_calls[0] >= max_calls or calls >= budget - 3 or time_left() < 1.0:
                        raise RuntimeError("budget exhausted")
                    x = np.asarray(x, dtype=np.float64)
                    fx = float(func(x))
                    calls += 1
                    nm_calls[0] += 1
                    all_points.append((fx, x.copy()))
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx
                result = minimize(
                    wrapped_func, start_x, method='Nelder-Mead',
                    options={
                        'maxiter': max_calls,
                        'xatol': 1e-9,
                        'fatol': 1e-11,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 1: Global exploration using Sobol quasi-random sequence
        # Sobol gives much better coverage than the broken LCG approach
        n_global = 150  # Spend 150 calls on global exploration
        if dim <= 32:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, scramble=True)
            # Generate points in [0,1)^dim
            n = n_global
            # Sobol requires power-of-2 for best quality, but works for any n
            sobol_points = sobol.random(n)
            for i in range(n):
                if calls >= budget - 100 or time_left() < 5.0:
                    break
                x = -5.0 + 10.0 * sobol_points[i]
                evaluate(x)
        else:
            # Fallback to random if Sobol unavailable
            for i in range(n_global):
                if calls >= budget - 100 or time_left() < 5.0:
                    break
                x = np.random.uniform(-5, 5, dim)
                evaluate(x)

        # Also evaluate a few structured points for coverage
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.array([5.0, -5.0, 0.0, 5.0, -5.0] if dim >= 5 else np.zeros(dim))[:dim],
            np.array([-5.0, 5.0, 0.0, -5.0, 5.0] if dim >= 5 else np.zeros(dim))[:dim],
        ]
        for p in structured:
            if calls >= budget - 100 or time_left() < 5.0:
                break
            evaluate(p)

        # Phase 2: Select top candidates from global exploration and refine
        # Sort by fitness and pick top diverse points
        if all_points and calls < budget - 50 and time_left() > 10.0:
            all_points.sort(key=lambda t: t[0])
            # Pick top 5 diverse starting points
            candidates = []
            min_dist = 1.0  # Minimum distance between candidates
            for fx, x in all_points[:50]:  # Look at top 50
                if len(candidates) >= 5:
                    break
                if all(np.linalg.norm(x - cx) > min_dist for _, cx in candidates):
                    candidates.append((fx, x))
            
            if not candidates:
                candidates = [(fx, x) for fx, x in all_points[:5]]
            
            # Refine each candidate with Nelder-Mead
            for fx, x in candidates:
                if calls >= budget - 30 or time_left() < 5.0:
                    break
                nm_minimize(x, max_calls=50)

        # Phase 3: Final polish on the best point found
        if best_x is not None and calls < budget - 10 and time_left() > 3.0:
            nm_minimize(best_x, max_calls=30)

        return best_x, best_f
