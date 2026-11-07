import numpy as np
import time
import scipy.optimize
import warnings

warnings.filterwarnings("ignore")


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)

        call_count = 0
        best_x = None
        best_fx = np.inf

        def time_ok():
            return time.time() < deadline

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        bounds = [(-5, 5)] * dim

        # Phase 1: Broad global exploration with structured + random points
        # Use a mix of deterministic structured points and random samples
        n_structured = 20
        n_random = 40

        candidates = []

        # Structured points: grid-like exploration
        for i in range(n_structured):
            # Use Latin hypercube-like sampling in [-4, 4]^5
            point = np.array([
                rng.uniform(-4, 4) for _ in range(dim)
            ])
            candidates.append(point)

        # Additional structured points
        candidates.append(np.zeros(dim))
        candidates.append(np.ones(dim))
        candidates.append(np.full(dim, 0.5))
        candidates.append(np.full(dim, -1.0))
        candidates.append(np.full(dim, 3.0))
        candidates.append(np.full(dim, -3.0))
        candidates.append(np.array([1, 1, 1, 1, 1], dtype=float))
        candidates.append(np.array([0, 0, 0, 0, 0], dtype=float))

        # Random points
        for _ in range(n_random):
            candidates.append(rng.uniform(-4.5, 4.5, dim))

        # Evaluate all candidates
        for x0 in candidates:
            if not time_ok() or call_count >= self.budget - 50:
                break
            try:
                tracked_func(x0)
            except TimeoutError:
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                tracked_func(best_x)
            except Exception:
                pass

        # Phase 2: Multi-start local optimization
        # Use the best few points from exploration as starting points
        # Sort by function value and pick top starts
        # We don't have stored evaluations, so use best and a few random offsets

        # Primary local optimization with L-BFGS-B from best point
        remaining_budget = self.budget - call_count - 20
        if time_ok() and remaining_budget > 20:
            try:
                scipy.optimize.minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 500,
                        'maxfun': max(10, remaining_budget // 2),
                        'ftol': 1e-14,
                        'gtol': 1e-14,
                        'eps': 1e-7
                    }
                )
            except TimeoutError:
                pass
            except Exception:
                pass

        # Phase 3: Second local optimization from a slightly perturbed start
        if time_ok() and call_count < self.budget - 10:
            remaining = self.budget - call_count - 5
            if remaining > 10:
                # Perturb best_x slightly and run another local optimization
                perturb = rng.normal(0, 0.1, dim)
                x2 = np.clip(best_x + perturb, -5, 5)
                try:
                    tracked_func(x2)
                    scipy.optimize.minimize(
                        tracked_func,
                        x2,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 300,
                            'maxfun': max(5, remaining // 2),
                            'ftol': 1e-14,
                            'gtol': 1e-14,
                            'eps': 1e-7
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        # Phase 4: Final polish with Nelder-Mead from best point
        if time_ok() and call_count < self.budget - 3:
            remaining = self.budget - call_count - 2
            if remaining > 5:
                try:
                    scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 200,
                            'maxfev': max(3, remaining),
                            'xatol': 1e-12,
                            'fatol': 1e-12
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        # Phase 5: Final local refinement with very small steps
        if time_ok() and call_count < self.budget - 2:
            remaining = self.budget - call_count - 1
            if remaining > 3:
                try:
                    scipy.optimize.minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 100,
                            'maxfun': max(2, remaining),
                            'ftol': 1e-16,
                            'gtol': 1e-16,
                            'eps': 1e-8
                        }
                    )
                except TimeoutError:
                    pass
                except Exception:
                    pass

        return best_x, best_fx
