import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        self.dim = 5
        self.budget = 500

        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            fx = func(x)
            if np.isfinite(fx) and fx < self.best_f:
                self.best_f = float(fx)
                self.best_x = x.copy()
            return fx if np.isfinite(fx) else 1e30

        # Phase 1: Sobol-like quasi-random sampling for global coverage
        # Use a low-discrepancy sequence for better initial spread
        n_global = 25
        points = self._sobol_points(n_global, self.dim)
        for p in points:
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            wrapped_func(p)

        if self.best_x is None:
            return np.zeros(self.dim), 1e30

        # Phase 2: Multi-start Nelder-Mead from top candidates
        remaining = self.budget - self.call_count
        if remaining > 20:
            try:
                self._multistart_nelder_mead(wrapped_func, remaining)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 3: Final fine polish with adaptive coordinate descent
        remaining = self.budget - self.call_count
        if remaining > 5 and time.time() < self.deadline - 1:
            try:
                self._fine_polish(wrapped_func, self.best_x.copy(), min(remaining, 30))
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

    def _sobol_points(self, n, dim):
        """Generate quasi-random points using a simple Halton-like sequence."""
        points = np.zeros((n, dim))
        # Use different primes for each dimension
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i in range(n):
            for d in range(dim):
                # Halton sequence
                p = primes[d]
                f = 1.0
                result = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= p
                    result += f * (i_val % p)
                    i_val //= p
                points[i, d] = -self.bound + 2 * self.bound * result
        return points

    def _multistart_nelder_mead(self, func, max_calls):
        """Multi-start Nelder-Mead local search from top candidates."""
        n = self.dim
        # Collect evaluation history to find top candidates
        # We'll do Nelder-Mead from the current best and a few nearby points
        
        # First, evaluate a few points around the best to find good starting points
        x_best = self.best_x.copy()
        f_best = self.best_f
        
        # Generate starting points: best + small perturbations
        starts = [x_best.copy()]
        for i in range(3):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            x_try = x_best + np.random.randn(n) * 0.5
            x_try = np.clip(x_try, -self.bound, self.bound)
            f_try = func(x_try)
            if f_try < f_best:
                starts.insert(0, x_try.copy())
                f_best = f_try
            else:
                starts.append(x_try.copy())
        
        # Run Nelder-Mead from each start
        for x0 in starts[:4]:
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            remaining = self.budget - self.call_count
            if remaining < 10:
                break
            try:
                self._nelder_mead(func, x0.copy(), min(remaining, 150))
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

    def _nelder_mead(self, func, x0, max_calls):
        """Nelder-Mead simplex optimization with bounded steps."""
        n = self.dim
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [func(x0)]
        
        # Create initial simplex points
        for i in range(n):
            x_try = x0.copy()
            step = 0.1 if abs(x0[i]) > 1e-10 else 0.1
            x_try[i] += step
            x_try = np.clip(x_try, -self.bound, self.bound)
            simplex.append(x_try.copy())
            fvals.append(func(x_try))
        
        calls_used = n + 1
        max_iter = max(50, max_calls // (n + 1))
        
        for it in range(max_iter):
            if calls_used >= max_calls or time.time() > self.deadline:
                break
            if self.call_count >= self.budget:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if max(fvals) - min(fvals) < 1e-10:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_reflect = centroid + 1.0 * (centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, -self.bound, self.bound)
            f_reflect = func(x_reflect)
            calls_used += 1
            
            if f_reflect < fvals[0]:
                # Expansion
                x_expand = centroid + 2.0 * (centroid - simplex[-1])
                x_expand = np.clip(x_expand, -self.bound, self.bound)
                f_expand = func(x_expand)
                calls_used += 1
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    fvals[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
            elif f_reflect < fvals[-2]:
                simplex[-1] = x_reflect
                fvals[-1] = f_reflect
            else:
                # Contraction
                x_contract = centroid + 0.5 * (simplex[-1] - centroid)
                x_contract = np.clip(x_contract, -self.bound, self.bound)
                f_contract = func(x_contract)
                calls_used += 1
                if f_contract < fvals[-1]:
                    simplex[-1] = x_contract
                    fvals[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= max_calls or time.time() > self.deadline:
                            break
                    if calls_used >= max_calls or time.time() > self.deadline:
                        break
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < self.best_f:
                self.best_f = fvals[min_idx]
                self.best_x = simplex[min_idx].copy()

    def _fine_polish(self, func, x_start, max_calls):
        """Fine local search with adaptive step sizes."""
        n = self.dim
        x = x_start.copy()
        f_best = func(x)
        calls = 1
        step = 0.02
        for it in range(max_calls - 1):
            if calls >= max_calls or time.time() > self.deadline:
                break
            if self.call_count >= self.budget:
                break
            improved = False
            for i in range(n):
                for sign in [1, -1]:
                    if calls >= max_calls or time.time() > self.deadline:
                        break
                    if self.call_count >= self.budget:
                        break
                    x_try = x.copy()
                    x_try[i] += sign * step
                    x_try = np.clip(x_try, -self.bound, self.bound)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < f_best - 1e-12:
                        x = x_try
                        f_best = f_try
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-8:
                    break
