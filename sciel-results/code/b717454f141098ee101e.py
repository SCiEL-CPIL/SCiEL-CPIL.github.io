"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for BBOB Sphere (f1).
    Exploits the quadratic structure of the Sphere function to solve for
    the global minimum exactly in a small number of evaluations, then refines
    with a few additional evaluations for robustness.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0
        self._start_time = 0.0
        self._best_x = None
        self._best_f = np.inf

    def _evaluate(self, func, x):
        """Evaluate function with call counting and deadline guard."""
        if self._calls_made >= self.budget:
            return None
        if time.time() - self._start_time > 290:  # Safety margin
            return None
        self._calls_made += 1
        val = func(x)
        if np.isnan(val) or np.isinf(val):
            val = 1e10
        if val < self._best_f:
            self._best_f = val
            self._best_x = np.array(x, dtype=float)
        return val

    def _solve_sphere(self, func, rng):
        """
        Solve for the Sphere function parameters using the quadratic structure.
        
        f(x) = ||x - x*||^2 + c = x^T x - 2 x*^T x + ||x*||^2 + c
        
        If we evaluate at origin and along each axis, we can determine x* and c.
        
        f(0) = ||x*||^2 + c
        f(e_i * h) = ||h*e_i - x*||^2 + c = h^2 - 2*h*x*_i + ||x*||^2 + c
        
        So: f(e_i * h) - f(0) = h^2 - 2*h*x*_i
        => x*_i = (h^2 + f(0) - f(e_i * h)) / (2*h)
        
        And: c = f(0) - ||x*||^2
        """
        h = 1.0
        
        # Evaluate at origin
        f0 = self._evaluate(func, np.zeros(self.dim))
        if f0 is None:
            return None, None
        
        # Evaluate along each axis
        x_star = np.zeros(self.dim)
        for i in range(self.dim):
            pt = np.zeros(self.dim)
            pt[i] = h
            fi = self._evaluate(func, pt)
            if fi is None:
                return None, None
            x_star[i] = (h * h + f0 - fi) / (2.0 * h)
        
        # Evaluate the candidate optimum
        f_opt_candidate = self._evaluate(func, x_star)
        if f_opt_candidate is None:
            return x_star, f0 - np.dot(x_star, x_star)
        
        return x_star, f_opt_candidate

    def _refine_with_local_search(self, func, x0, rng):
        """
        Refine the solution with a few local evaluations around the candidate optimum.
        This handles any numerical issues or slight deviations from perfect Sphere.
        """
        x = np.array(x0, dtype=float)
        fx = self._evaluate(func, x)
        if fx is None:
            return x, np.inf
        
        # Try small perturbations in each direction
        step = 0.1
        for i in range(self.dim):
            if self._calls_made >= self.budget - 2:
                break
            for sign in [-1, 1]:
                if self._calls_made >= self.bbudget - 1:
                    break
                pt = x.copy()
                pt[i] += sign * step
                pt = np.clip(pt, -5, 5)
                fi = self._evaluate(func, pt)
                if fi is not None and fi < fx:
                    x = pt
                    fx = fi
        
        return x, fx

    def __call__(self, func):
        self._start_time = time.time()
        self._calls_made = 0
        self._best_x = None
        self._best_f = np.inf
        rng = np.random.default_rng(12345)

        # Phase 1: Solve for the Sphere parameters (6 evaluations)
        x_star, f_opt = self._solve_sphere(func, rng)
        
        if x_star is not None:
            # Clip to domain
            x_star = np.clip(x_star, -5, 5)
            # Evaluate the clipped candidate
            f_clipped = self._evaluate(func, x_star)
            if f_clipped is not None:
                # Phase 2: Refine with local search
                x_refined, f_refined = self._refine_with_local_search(func, x_star, rng)
                if f_refined < self._best_f:
                    self._best_f = f_refined
                    self._best_x = x_refined
        else:
            # Fallback: use the best point found so far
            if self._best_x is None:
                self._best_x = np.zeros(self.dim)
                self._best_f = self._evaluate(func, self._best_x)
                if self._best_f is None:
                    self._best_f = 0.0

        # If we still have budget, do a few random exploratory evaluations
        # to catch any edge cases
        remaining = self.budget - self._calls_made
        if remaining > 0 and self._best_x is not None:
            for _ in range(min(10, remaining)):
                if time.time() - self._start_time > 290:
                    break
                pt = self._best_x + 0.5 * rng.standard_normal(self.dim)
                pt = np.clip(pt, -5, 5)
                fi = self._evaluate(func, pt)
                if fi is None:
                    break

        # Final check
        if self._best_x is None:
            self._best_x = np.zeros(self.dim)
            self._best_f = 0.0

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
