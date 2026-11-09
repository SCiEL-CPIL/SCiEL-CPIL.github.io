"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator


class Optimizer:
    """
    Surrogate-assisted search combining:
    - Initial random sampling to build a model
    - RBF surrogate for cheap predictions
    - Local refinement using scipy minimize (Nelder-Mead) around promising points
    - Iterative refinement with budget management
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def _count_call(self, func, x):
        """Evaluate func(x) and track call count."""
        self.calls += 1
        return func(x)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls = 0

        # Phase 1: Initial random sampling to build initial model
        n_init = min(20, budget)
        rng_init = np.random.default_rng(42)
        X = rng_init.uniform(-5, 5, size=(n_init, dim))
        F = np.array([self._count_call(func, x) for x in X])

        best_idx = np.argmin(F)
        x_best = X[best_idx].copy()
        f_best = F[best_idx]

        # Phase 2: Iterative surrogate-guided search
        remaining = budget - self.calls
        n_iter = 0
        rng = np.random.default_rng(123)
        rbf = None
        last_update_iter = -10

        while remaining > 0 and n_iter < 300:
            # Rebuild RBF surrogate periodically or when significantly improved
            if n_iter - last_update_iter >= 5 or rbf is None:
                if len(X) >= 3:
                    try:
                        rbf = RBFInterpolator(X, F, kernel='thin_plate_spline')
                        last_update_iter = n_iter
                    except Exception:
                        try:
                            rbf = RBFInterpolator(X, F, kernel='linear')
                            last_update_iter = n_iter
                        except Exception:
                            rbf = None

            # Strategy: alternate between exploitation and exploration
            # Use exploitation when we have a good surrogate, else explore more
            
            if rbf is not None and remaining >= 2:
                # Exploitation: use surrogate to guide search
                # 1. Local refinement around best point
                x_cand = None
                try:
                    def surrogate_local(x):
                        return rbf(x.reshape(1, -1))[0]

                    res = minimize(
                        surrogate_local,
                        x_best,
                        method='Nelder-Mead',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 20, 'xatol': 1e-5, 'fatol': 1e-10}
                    )
                    x_cand = np.clip(res.x, -5, 5)
                except Exception:
                    pass

                if x_cand is None:
                    # Fallback: random perturbation
                    x_cand = x_best + 0.5 * rng.standard_normal(dim)
                    x_cand = np.clip(x_cand, -5, 5)
            else:
                # Exploration: pure random or large perturbations
                if rng.random() < 0.5:
                    # Pure random
                    x_cand = rng.uniform(-5, 5, dim)
                else:
                    # Large perturbation around best
                    x_cand = x_best + 2.0 * rng.standard_normal(dim)
                    x_cand = np.clip(x_cand, -5, 5)

            # Evaluate the candidate
            f_cand = self._count_call(func, x_cand)

            # Update model
            X = np.vstack([X, x_cand.reshape(1, -1)])
            F = np.append(F, f_cand)

            if f_cand < f_best:
                f_best = f_cand
                x_best = x_cand.copy()

            remaining = budget - self.calls
            n_iter += 1

        return x_best, f_best


# EVOLVE-BLOCK-END
