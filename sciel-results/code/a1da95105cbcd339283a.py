import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 310.0

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()

        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(np.clip(x, -5.0, 5.0))

        n = self.dim
        best_x = np.zeros(n)
        best_fx = float('inf')

        def nelder_mead(x0, max_iters, ftol=1e-10, xatol=1e-8):
            """Nelder-Mead with adaptive restarts for anisotropic functions."""
            nonlocal best_x, best_fx
            x = x0.copy()
            f = eval_func(x)
            if f < best_fx:
                best_fx = f
                best_x = x.copy()

            # Initialize simplex
            simplex = [x.copy()]
            for i in range(n):
                xi = x.copy()
                step = 0.25 if x[i] == 0 else 0.05 * abs(x[i])
                xi[i] += step
                simplex.append(xi)

            fvals = [f]
            for i in range(1, n + 1):
                fvals.append(eval_func(simplex[i]))
                if fvals[-1] < best_fx:
                    best_fx = fvals[-1]
                    best_x = simplex[i].copy()

            for it in range(max_iters):
                if self.calls >= self.budget:
                    break
                if time.time() - self.start_time > self.time_limit:
                    break

                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                if abs(fvals[-1] - fvals[0]) < ftol:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                fr = eval_func(xr)
                if fr < best_fx:
                    best_fx = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    fe = eval_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = eval_func(simplex[i])
                            if fvals[i] < best_fx:
                                best_fx = fvals[i]
                                best_x = simplex[i].copy()

                # Check convergence
                if max(fvals) - min(fvals) < ftol:
                    break

            return best_x, best_fx

        def coordinate_descent(x0, max_iters, tol=1e-10):
            """Adaptive coordinate descent for stiff dimensions."""
            nonlocal best_x, best_fx
            x = x0.copy()
            f = eval_func(x)
            if f < best_fx:
                best_fx = f
                best_x = x.copy()

            step_sizes = np.ones(n) * 0.1

            for it in range(max_iters):
                if self.calls >= self.budget:
                    break
                if time.time() - self.start_time > self.time_limit:
                    break

                improved = False
                for i in range(n):
                    if self.calls >= self.budget:
                        break
                    if time.time() - self.start_time > self.time_limit:
                        break

                    # Try positive step
                    x_trial = x.copy()
                    x_trial[i] += step_sizes[i]
                    f_trial = eval_func(x_trial)
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        if f < best_fx:
                            best_fx = f
                            best_x = x.copy()
                        # Try to expand
                        for _ in range(3):
                            x_trial2 = x.copy()
                            x_trial2[i] += step_sizes[i]
                            f_trial2 = eval_func(x_trial2)
                            if f_trial2 < f:
                                x = x_trial2
                                f = f_trial2
                                step_sizes[i] *= 1.5
                                if f < best_fx:
                                    best_fx = f
                                    best_x = x.copy()
                            else:
                                break
                    else:
                        # Try negative step
                        x_trial = x.copy()
                        x_trial[i] -= step_sizes[i]
                        f_trial = eval_func(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            if f < best_fx:
                                best_fx = f
                                best_x = x.copy()
                            # Try to expand
                            for _ in range(3):
                                x_trial2 = x.copy()
                                x_trial2[i] -= step_sizes[i]
                                f_trial2 = eval_func(x_trial2)
                                if f_trial2 < f:
                                    x = x_trial2
                                    f = f_trial2
                                    step_sizes[i] *= 1.5
                                    if f < best_fx:
                                        best_fx = f
                                        best_x = x.copy()
                                else:
                                    break
                        else:
                            step_sizes[i] *= 0.5

                if not improved:
                    break

            return best_x, best_fx

        rng = np.random.default_rng(42)

        # Phase 1: Initial random evaluations to find good starting points
        n_initial = min(10, self.budget // 10)
        candidates = [np.zeros(n)]
        for _ in range(n_initial - 1):
            candidates.append(rng.uniform(-4, 4, n))

        for x0 in candidates:
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
            try:
                f = eval_func(x0)
                if f < best_fx:
                    best_fx = f
                    best_x = x0.copy()
            except StopIteration:
                break

        # Phase 2: Nelder-Mead from best point
        remaining = self.budget - self.calls
        if remaining > 20 and time.time() - self.start_time < self.time_limit - 5:
            try:
                max_iters = min(100, remaining // (n + 1))
                nelder_mead(best_x.copy(), max_iters)
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Coordinate descent polish
        remaining = self.budget - self.calls
        if remaining > 10 and time.time() - self.start_time < self.time_limit - 5:
            try:
                max_iters = min(50, remaining // (n + 1))
                coordinate_descent(best_x.copy(), max_iters)
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 4: Final Nelder-Mead polish with tighter tolerance
        remaining = self.budget - self.calls
        if remaining > 5 and time.time() - self.start_time < self.time_limit - 3:
            try:
                max_iters = min(30, remaining // (n + 1))
                nelder_mead(best_x.copy(), max_iters, ftol=1e-12, xatol=1e-10)
            except StopIteration:
                pass
            except Exception:
                pass

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(n)
            if self.calls < self.budget:
                best_fx = eval_func(best_x)
            else:
                best_fx = 0.0

        return best_x, best_fx
