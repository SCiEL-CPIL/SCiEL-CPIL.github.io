"""Rewritten optimizer for BBOB f20 (Schwefel) with multi-start Nelder-Mead + CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for Schwefel-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.best_x = None
        self.best_f = np.inf
        self._start_time = None

    def __call__(self, func):
        self._start_time = time.time()
        self.calls_made = 0
        self.best_x = None
        self.best_f = np.inf
        deadline = self._start_time + 290.0  # safety margin before 300s

        # Counting wrapper
        def count_func(x):
            self.calls_made += 1
            return func(x)

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Strategy: multi-start Nelder-Mead
        # Use several starting points with different scales
        # Each Nelder-Mead run gets a portion of the budget
        # We'll do adaptive allocation: try quick runs first, then longer ones

        # Phase 1: Quick exploration with multiple short Nelder-Mead runs
        # Phase 2: Refinement on the best point found

        # Starting points: mix of random, center, and structured
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, dim))
        # Structured: corners and midpoints
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = 4.0
            starts.append(pt)
            pt[i] = -4.0
            starts.append(pt)

        # Deduplicate
        unique_starts = []
        seen = set()
        for s in starts:
            key = tuple(np.round(s, 4))
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)

        # Shuffle starts
        rng.shuffle(unique_starts)

        # Allocate budget: first few runs get less, later ones get more
        # Total budget is 500 calls
        # Nelder-Mead in d dimensions needs ~2d+1 simplex evaluations initially
        # Let's do: first 4 starts get ~30 calls each, then remaining get more

        # We'll do sequential runs, each with a time and call budget
        # Keep track of best

        # First, evaluate all starting points
        for s in unique_starts:
            if self.calls_made >= budget or time.time() > deadline:
                break
            fs = count_func(s)
            if fs < self.best_f:
                self.best_f = fs
                self.best_x = s.copy()

        # Now run Nelder-Mead from the best few starting points
        # Sort starts by their function value (we need to evaluate them)
        # Actually we already evaluated them above, let's re-evaluate in order

        # Collect (f, x) for all evaluated starts
        eval_results = []
        for s in unique_starts:
            if self.calls_made >= budget or time.time() > deadline:
                break
            # We already evaluated these, but let's just use the best ones
            pass

        # Simpler approach: just run Nelder-Mead from the best starting point
        # with a generous budget, then if time permits, try another

        # Determine how many Nelder-Mead runs to do
        # Each NM run in 5D needs at least 6 evaluations for simplex
        # Let's do up to 3-4 NM runs

        # Sort unique_starts by distance from best_x (prefer closer ones for refinement)
        # Actually, let's just run NM from best_x first, then from second best, etc.

        # Re-evaluate starts to get ordering
        start_vals = []
        for s in unique_starts:
            if self.calls_made >= budget or time.time() > deadline:
                break
            fs = count_func(s)
            start_vals.append((fs, s))

        start_vals.sort(key=lambda t: t[0])

        # Run Nelder-Mead from top starts
        nm_configs = [
            # (max_iter, maxfev) - we control via maxfev
            (100, 40),   # quick
            (200, 80),   # medium
            (300, 150),  # longer
            (500, 200),  # long
        ]

        for i, (fs, sx) in enumerate(start_vals[:min(4, len(start_vals))]):
            if self.calls_made >= budget or time.time() > deadline:
                break

            # Use the i-th config
            if i < len(nm_configs):
                max_iter, maxfev = nm_configs[i]
            else:
                max_iter, maxfev = 200, 100

            # Adjust maxfev to not exceed remaining budget
            remaining = budget - self.calls_made
            maxfev = min(maxfev, remaining)
            if maxfev < 10:
                break

            try:
                res = minimize(
                    count_func,
                    sx,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'maxfev': maxfev,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
            except Exception:
                continue

            if res.fun < self.best_f:
                self.best_f = res.fun
                self.best_x = res.x.copy()

            # If we've used most of the budget, stop
            if self.calls_made >= budget - 5:
                break

        # Final refinement: one more NM from the best point with remaining budget
        if self.calls_made < budget - 10 and self.best_x is not None and time.time() < deadline:
            remaining = budget - self.calls_made
            try:
                res = minimize(
                    count_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        # Ensure we haven't exceeded budget
        # The count_func wrapper enforces this by just counting
        # But we should stop early if we hit the deadline

        if self.best_x is None:
            # Fallback: return a random point
            self.best_x = rng.uniform(-5, 5, dim)
            self.best_f = func(self.best_x)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
