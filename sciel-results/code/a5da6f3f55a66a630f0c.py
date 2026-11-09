"""Rewritten optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Sobol + BFGS local search for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0  # safety margin before 300s kill
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        # Sobol sequence for low-discrepancy initialization
        # Use scipy's Sobol quasi-random sequence
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        except ImportError:
            # Fallback: use Halton-like sequence
            class _SobolFallback:
                def __init__(self, d, seed=0):
                    self.d = d
                    self._rng = np.random.default_rng(seed)
                    self._count = 0
                def random(self, m):
                    self._count += m
                    return self._rng.random((m, self.d))
            sobol = _SobolFallback(n)

        # Generate initial points from Sobol sequence mapped to [-5, 5]^n
        # We'll use batches to spread out evaluations
        # First batch: broad exploration
        n_init = min(64, budget // 4)  # use up to 64 initial points
        sobol.random(1)  # skip first point (origin)
        points = sobol.random(n_init) * 10.0 - 5.0  # map to [-5, 5]

        best_x = None
        best_f = np.inf
        calls = 0

        # Evaluate initial Sobol points
        for i in range(n_init):
            if calls >= budget or elapsed() > deadline:
                break
            x = points[i]
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Now do sequential BFGS refinement from the best points
        # Keep top-k candidates for refinement
        # Re-evaluate to get sorted list (we already have them, but let's track)
        # Actually we only tracked the best; let's do a second pass with more structure

        # Strategy: from each of the top few initial points, run BFGS
        # We need to track all initial evaluations
        # Let's redo: collect all initial (x, f) pairs
        # Since we already evaluated, let's just use the best and a few others

        # Actually, let's be more systematic:
        # Phase 1: Evaluate Sobol points (done above)
        # Phase 2: For the best few points, run BFGS to refine

        # Get top candidates from initial phase
        # We didn't store all of them, so let's just use the best and generate
        # a few more diverse starting points

        # Let's do a second round of Sobol points in a smaller region around best
        # Then BFGS from the overall best

        # BFGS refinement
        if best_x is not None and calls < budget and elapsed() < deadline:
            # Run BFGS from best_x
            def objective(x):
                nonlocal calls
                if calls >= budget:
                    return best_f  # return current best to stop
                f = func(x)
                calls += 1
                return f

            try:
                result = minimize(
                    objective,
                    best_x,
                    method='BFGS',
                    options={'maxiter': 50, 'gtol': 1e-6}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # If we still have budget, do multi-start BFGS from several good points
        # Generate more Sobol points in the region around the best
        if calls < budget - 10 and elapsed() < deadline:
            # Generate additional starting points near the best
            n_extra_starts = min(8, (budget - calls) // 20)
            rng = np.random.default_rng(123)
            for s in range(n_extra_starts):
                if calls >= budget or elapsed() > deadline:
                    break
                # Start near best with some perturbation
                scale = 2.0 / (s + 1)
                x0 = best_x + rng.standard_normal(n) * scale
                x0 = np.clip(x0, -5.0, 5.0)

                f0 = func(x0)
                calls += 1
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()

                # BFGS from this point
                if calls < budget - 5 and elapsed() < deadline:
                    def objective2(x):
                        nonlocal calls
                        if calls >= budget:
                            return best_f
                        f = func(x)
                        calls += 1
                        return f

                    try:
                        result = minimize(
                            objective2,
                            x0,
                            method='BFGS',
                            options={'maxiter': 30, 'gtol': 1e-6}
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x.copy()
                    except Exception:
                        pass

        # Final: if budget remains, do a simple coordinate descent / Nelder-Mead
        if calls < budget - 5 and elapsed() < deadline:
            def objective3(x):
                nonlocal calls
                if calls >= budget:
                    return best_f
                f = func(x)
                calls += 1
                return f

            try:
                result = minimize(
                    objective3,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
