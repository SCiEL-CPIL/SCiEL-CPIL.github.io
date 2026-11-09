"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start (1+1)-ES for multimodal Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin below 300 s
        n = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(2024)

        # Build a diverse set of starting points (all distinct, in-box)
        starts = []
        starts.append(np.zeros(n))
        # Halton-style quasi-random low-discrepancy points
        def halton(idx, base):
            f, r = 1.0, 0.0
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            return r
        for k in range(1, 40):
            pt = np.array([halton(k, b) for b in range(2, n + 2)])
            starts.append(lo + pt * (hi - lo))
        # Random points to fill up to 48 starts
        while len(starts) < 48:
            starts.append(rng.uniform(lo, hi, n))

        # Allocate budget across starts: more budget to earlier (better) starts
        # Use a schedule: each start gets a minimum, remaining grows with progress.
        # Simpler: give each start a per-run budget that shrinks as we go,
        # but ensure at least a few calls per start for coverage.
        n_starts = len(starts)
        # Per-run budget schedule: front-loaded so early promising runs get more.
        # We'll decide dynamically: keep a base per-run budget and shrink over time.
        base_run = 20
        min_run = 4

        best_x = None
        best_f = np.inf
        calls_used = 0

        def run_es(x0, fx, run_budget):
            nonlocal calls_used
            x = x0.copy()
            f = fx
            sigma = 1.5
            # 1/5 success rule state
            window = 0
            successes = 0
            stagnant = 0
            for _ in range(run_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), lo, hi)
                fy = func(y)
                calls_used += 1
                if fy < f:
                    x, f = y, fy
                    successes += 1
                    stagnant = 0
                else:
                    stagnant += 1
                window += 1
                if window >= 10:
                    # 1/5 rule: scale sigma by success rate
                    rate = successes / window
                    if rate > 0.2:
                        sigma *= 1.25
                    elif rate < 0.2:
                        sigma *= 0.8
                    window = 0
                    successes = 0
                # Clamp sigma
                sigma = min(max(sigma, 1e-3), 8.0)
                # Restart within run if long stagnant
                if stagnant > 25:
                    x = x + 0.5 * rng.standard_normal(n)
                    x = np.clip(x, lo, hi)
                    fx2 = func(x)
                    calls_used += 1
                    f = fx2
                    stagnant = 0
                    sigma = 1.0
            return x, f

        for i, s in enumerate(starts):
            if calls_used >= budget or time.time() > deadline:
                break
            # Per-run budget: front-loaded, shrinking as we consume budget
            remaining = budget - calls_used
            # Give later starts less; front starts more.
            frac = (n_starts - i) / n_starts
            run_budget = max(min_run, int(base_run * frac) + 2)
            # Cap by remaining
            run_budget = min(run_budget, remaining)
            if run_budget < 1:
                continue
            # Evaluate start
            fs = func(s)
            calls_used += 1
            if fs < best_f:
                best_f, best_x = fs, s.copy()
            # Run ES from this start
            x, f = run_es(s, fs, run_budget - 1)
            if f < best_f:
                best_f, best_x = f, x.copy()

        # If we still have budget and time, polish the best point with a
        # final (1+1)-ES using the remaining calls.
        if best_x is not None and calls_used < budget:
            x, f = run_es(best_x, best_f, budget - calls_used)
            if f < best_f:
                best_f, best_x = f, x.copy()

        if best_x is None:
            best_x = np.zeros(n)
        return best_x, best_f
# EVOLVE-BLOCK-END
