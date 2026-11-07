"""Rewritten optimizer: multi-start Nelder-Mead with adaptive budget allocation."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead for low-dimensional multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Track best point and value
        best_x = None
        best_f = np.inf
        eval_count = 0

        # Wrap func to count evaluations and enforce deadline
        def wrapped_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError()
            if eval_count >= budget:
                raise ValueError("Budget exceeded")
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            eval_count += 1
            try:
                fval = float(func(x))
            except Exception:
                fval = np.inf
            if np.isfinite(fval) and fval < best_f:
                best_f = fval
                best_x = x.copy()
            return fval

        # Generate diverse starting points
        # Use a combination of random points and some structured points
        n_starts = 30  # Number of initial starting points

        # First, try the origin and a few symmetric points
        starts = []
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim) * 3.0)
        starts.append(np.ones(dim) * -3.0)
        starts.append(np.ones(dim) * 1.5)
        starts.append(np.ones(dim) * -1.5)

        # Random starts
        for _ in range(n_starts - len(starts)):
            starts.append(rng.uniform(-5.0, 5.0, dim))

        # Shuffle starts to randomize order
        indices = rng.permutation(len(starts))
        starts = [starts[i] for i in indices]

        # Nelder-Mead options
        # maxiter controls the number of iterations in each local optimization
        # We need to be careful about total budget
        # Each Nelder-Mead call uses roughly (dim+1) * maxiter evaluations
        # With dim=5, that's 6 * maxiter per start
        # If we use all budget for 30 starts, each gets ~16 evaluations, so maxiter ~2
        # Better approach: allocate budget adaptively

        # Strategy: 
        # 1. Do a quick pass over all starts with small maxiter to find promising regions
        # 2. Then do deeper optimization on the best few starting points

        # Phase 1: Quick exploration
        quick_maxiter = 5  # Small number of iterations per start in phase 1
        phase1_budget_per_start = max(1, (budget // 2) // n_starts)

        for i, x0 in enumerate(starts):
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': quick_maxiter,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # Phase 2: Deep optimization on best starting points
        # Use the best few points from phase 1 as new starts
        # Re-initialize best tracking for phase 2 starts
        # We'll use the best_x from phase 1 and a few neighbors

        # Collect candidate starts for phase 2
        phase2_starts = []
        if best_x is not None:
            phase2_starts.append(best_x.copy())
            # Add some perturbations around the best point
            for _ in range(5):
                perturbation = rng.normal(0, 1.0, dim)
                new_x = np.clip(best_x + perturbation, -5.0, 5.0)
                phase2_starts.append(new_x)

        # Also add the original starts that gave good results
        # We don't track per-start results, so just use random additional starts
        for _ in range(10):
            phase2_starts.append(rng.uniform(-5.0, 5.0, dim))

        # Shuffle phase 2 starts
        indices2 = rng.permutation(len(phase2_starts))
        phase2_starts = [phase2_starts[i] for i in indices2]

        # Phase 2: deeper optimization
        remaining_budget = budget - eval_count
        if remaining_budget > 0:
            n_phase2 = len(phase2_starts)
            deep_maxiter = max(10, remaining_budget // (6 * n_phase2))  # Rough estimate

            for i, x0 in enumerate(phase2_starts):
                if time.time() > deadline or eval_count >= budget:
                    break
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': deep_maxiter,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    continue

        # If we still have budget, do more random restarts with moderate depth
        while eval_count < budget and time.time() < deadline:
            x0 = rng.uniform(-5.0, 5.0, dim)
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 20,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
