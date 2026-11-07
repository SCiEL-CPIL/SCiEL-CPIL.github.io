"""Rewritten optimizer: Sobol space-filling global search + Nelder-Mead local exploitation for Lunacek f24."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Deterministic Sobol-space-filling global search followed by Nelder-Mead simplex local exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = None

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 280.0  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return 1e12
            if time.time() > self._deadline:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        best_x = None
        best_f = np.inf

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ============================================================
        # Phase 1: Deterministic Sobol space-filling global search
        # Use a quasi-random Sobol sequence to densely and uniformly
        # map the bi-modal landscape, identifying the deep basin.
        # ============================================================
        n_sobol = 128  # 128 Sobol points for dense space-filling
        # Generate Sobol sequence in [0,1]^dim, then map to [-5,5]^dim
        # scipy.stats.qmc.Sobol is available in scipy >= 1.7
        try:
            from scipy.stats.qmc import Sobol
            sobol = Sobol(d=dim, scramble=True, seed=42)
            # Need power-of-2 number of points for Sobol
            n_sobol = 128
            unit_samples = sobol.random(n_sobol)  # shape (128, dim) in [0,1]
            sobol_points = unit_samples * 10.0 - 5.0  # map to [-5,5]
        except Exception:
            # Fallback: Halton-like deterministic sequence
            n_sobol = 128
            sobol_points = np.zeros((n_sobol, dim))
            # Use a simple deterministic low-discrepancy construction
            # Van der Corput sequence in each dimension with different bases
            def van_der_corput(n, base):
                result = 0.0
                f = 1.0 / base
                while n > 0:
                    result += f * (n % base)
                    n //= base
                    f /= base
                return result

            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  # first 10 primes
            for i in range(n_sobol):
                for d in range(dim):
                    sobol_points[i, d] = van_der_corput(i + 1, bases[d % len(bases)])
            sobol_points = sobol_points * 10.0 - 5.0

        # Evaluate Sobol points
        for x0 in sobol_points:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            fx0 = f(x0)
            update_best(x0, fx0)

        # Add a few strategic points if budget allows
        strategic = [
            np.zeros(dim),
            np.full(dim, 2.5),
            np.full(dim, -2.5),
            np.full(dim, 1.5),
            np.full(dim, -1.5),
        ]
        for x0 in strategic:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            fx0 = f(x0)
            update_best(x0, fx0)

        # ============================================================
        # Phase 2: Nelder-Mead simplex local exploitation
        # From the best global point, use Nelder-Mead to converge
        # precisely within the identified basin.
        # ============================================================
        if best_x is not None and calls[0] < budget and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # ============================================================
        # Phase 3: If budget remains, run Nelder-Mead from top-3
        # Sobol points to exploit multiple basins
        # ============================================================
        if calls[0] < budget and time.time() < self._deadline:
            # Re-evaluate top candidates from Sobol set
            # We'll pick the best 3 distinct regions
            if best_x is not None:
                # Try Nelder-Mead from a few nearby perturbed points
                perturbations = [
                    best_x + 0.5 * np.array([1, -1, 1, -1, 1][:dim]),
                    best_x + 0.5 * np.array([-1, 1, -1, 1, -1][:dim]),
                    best_x + 0.3 * np.array([1, 1, -1, -1, 1][:dim]),
                ]
                for x0 in perturbations:
                    if calls[0] >= budget or time.time() > self._deadline:
                        break
                    # Clip to bounds
                    x0 = np.clip(x0, -5.0, 5.0)
                    try:
                        res = minimize(
                            f,
                            x0,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 100,
                                'maxfev': 50,
                                'xatol': 1e-10,
                                'fatol': 1e-12,
                                'adaptive': True,
                            }
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # ============================================================
        # Phase 4: Final L-BFGS-B polish for precise convergence
        # ============================================================
        if best_x is not None and calls[0] < budget and time.time() < self._deadline:
            try:
                res = minimize(
                    f,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxfun': 30, 'ftol': 1e-15}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # ============================================================
        # Phase 5: If budget still remains, do a second Sobol sweep
        # with offset to catch any missed basin
        # ============================================================
        remaining = budget - calls[0]
        if remaining > 40 and time.time() < self._deadline:
            try:
                from scipy.stats.qmc import Sobol
                n_extra = min(64, remaining - 20)
                sobol2 = Sobol(d=dim, scramble=True, seed=123)
                unit_samples2 = sobol2.random(n_extra)
                sobol_points2 = unit_samples2 * 10.0 - 5.0
                for x0 in sobol_points2:
                    if calls[0] >= budget or time.time() > self._deadline:
                        break
                    fx0 = f(x0)
                    update_best(x0, fx0)

                # Quick Nelder-Mead from new best
                if best_x is not None and calls[0] < budget and time.time() < self._deadline:
                    res = minimize(
                        f,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'maxfev': 30,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
