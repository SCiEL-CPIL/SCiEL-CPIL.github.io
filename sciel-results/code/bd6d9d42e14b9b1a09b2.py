"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: multi-start with Nelder-Mead local polishing and (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            calls_made += 1
            return func(x)

        # Strategy: multi-start with Nelder-Mead local refinement
        # For Rosenbrock, local methods from good starts work well
        # We'll try several starting points and polish each with Nelder-Mead

        # Initial diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Random points
        for _ in range(19):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Shuffle starts to get diversity early
        rng.shuffle(starts)

        # Phase 1: Quick evaluation of all starts
        for x0 in starts:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            try:
                fx = safe_func(x0)
                if fx < best_f:
                    best_f = fx
                    best_x = x0.copy()
            except Exception:
                pass

        if best_x is None:
            # Fallback: just do a simple (1+1)-ES
            x = rng.uniform(-5, 5, self.dim)
            try:
                fx = safe_func(x)
            except Exception:
                return None, np.inf
            best_x, best_f = x, fx
            sigma = 1.0
            while calls_made < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                try:
                    fy = safe_func(y)
                except Exception:
                    break
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9

            return best_x, best_f

        # Phase 2: Polish the best starts with Nelder-Mead
        # Sort starts by their function values (we need to re-evaluate or track)
        # We'll polish the top few candidates
        candidates = [(best_x.copy(), best_f)]

        # Add some more diverse candidates for polishing
        for x0 in starts[:10]:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break
            try:
                fx = safe_func(x0)
                if fx < best_f:
                    best_f = fx
                    best_x = x0.copy()
                    candidates = [(best_x.copy(), best_f)]
                else:
                    candidates.append((x0.copy(), fx))
            except Exception:
                pass

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[1])

        # Polish top candidates with Nelder-Mead
        for x0, f0 in candidates[:8]:
            if calls_made >= self.budget:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: Fine-tune with (1+1)-ES from the best point
        if calls_made < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(50):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                try:
                    fy = safe_func(y)
                except Exception:
                    break
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
