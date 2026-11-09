"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import dual_annealing, differential_evolution, minimize


class Optimizer:
    """Portfolio of global optimizers with explicit deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._eval_count = 0
        self._start_time = None
        self._best_x = None
        self._best_f = np.inf

    def _record(self, x, fx):
        """Record an evaluation and update best."""
        self._eval_count += 1
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = x.copy()

    def _make_func(self, func):
        """Wrap func to track evaluations."""
        def wrapped(x):
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            fx = float(func(x))
            self._record(x, fx)
            return fx
        return wrapped

    def __call__(self, func):
        self._start_time = time.time()
        deadline = 300.0  # safety margin before 330s kill

        # Initial random evaluation to have a valid first call
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-4, 4, self.dim)
        f0 = float(func(x0))
        self._record(x0, f0)

        if self._eval_count >= self.budget:
            return self._best_x, self._best_f

        # Build wrapped function
        wfunc = self._make_func(func)

        # --- Phase 1: Differential Evolution (global, robust for multimodal) ---
        # Weierstrass is highly multimodal; DE is good at exploring.
        # Limit function evaluations to stay within budget.
        max_de_evals = min(self.budget - self._eval_count, 200)
        if max_de_evals > 10:
            try:
                bounds = [(-5.0, 5.0)] * self.dim
                de_res = differential_evolution(
                    wfunc,
                    bounds=bounds,
                    maxiter=50,
                    tol=1e-10,
                    popsize=15,
                    mutation=(0.3, 1.2),
                    recombination=0.9,
                    seed=42,
                    polish=False,  # We'll do local refinement ourselves
                    init='latinhypercube'
                )
            except Exception:
                pass

        if self._eval_count >= self.budget or (time.time() - self._start_time) > deadline:
            return self._best_x, self._best_f

        # --- Phase 2: Dual Annealing (multi-start simulated annealing) ---
        # Good for escaping local minima in Weierstrass-like functions.
        remaining = self.budget - self._eval_count
        if remaining > 20:
            try:
                da_res = dual_annealing(
                    wfunc,
                    bounds=[(-5.0, 5.0)] * self.dim,
                    seed=42,
                    maxiter=100,
                    initial_temp=5.23,
                    restart_temp_ratio=2e-5,
                    visit=2.62,
                    acceptance=2.07,
                    maxfun=remaining
                )
            except Exception:
                pass

        if self._eval_count >= self.budget or (time.time() - self._start_time) > deadline:
            return self._best_x, self._best_f

        # --- Phase 3: Local polish with Nelder-Mead around best point ---
        remaining = self.budget - self._eval_count
        if remaining > 10 and self._best_x is not None:
            try:
                res = minimize(
                    wfunc,
                    x0=self._best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass

        # Final check
        if self._eval_count >= self.budget or (time.time() - self._start_time) > deadline:
            return self._best_x, self._best_f

        # --- Phase 4: If budget remains, do a few more DE iterations or random restarts ---
        remaining = self.budget - self._eval_count
        if remaining > 20:
            try:
                de_res2 = differential_evolution(
                    wfunc,
                    bounds=[(-5.0, 5.0)] * self.dim,
                    maxiter=30,
                    tol=1e-12,
                    popsize=10,
                    mutation=(0.5, 1.0),
                    recombination=0.8,
                    seed=123,
                    polish=False,
                    init='sobol'
                )
            except Exception:
                pass

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
