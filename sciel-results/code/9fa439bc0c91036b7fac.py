import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead + Coordinate Descent optimizer for f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s limit
        calls = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse initial population to find a good starting region
        rng = np.random.default_rng(42)
        n_init = min(50, budget // 5)
        candidates = []
        for i in range(n_init):
            if calls >= budget - 10 or time.time() >= deadline:
                break
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.array([5.0] * dim)
            elif i == 2:
                x0 = np.array([-5.0] * dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            f = call_func(x0)
            candidates.append((f, x0.copy()))

        candidates.sort(key=lambda t: t[0])
        best_f = candidates[0][0]
        best_x = candidates[0][1].copy()

        # Phase 2: Nelder-Mead from the best starting point
        if calls < budget - 10 and time.time() < deadline:
            best_x, best_f = self._nelder_mead(func, call_func, best_x, best_f, dim, budget, deadline)

        # Phase 3: Coordinate descent refinement (exploit ridge structure)
        if calls < budget - 5 and time.time() < deadline:
            best_x, best_f = self._coordinate_descent(func, call_func, best_x, best_f, dim, budget, deadline)

        # Phase 4: Final Nelder-Mead polish
        if calls < budget - 1 and time.time() < deadline:
            best_x, best_f = self._nelder_mead(func, call_func, best_x, best_f, dim, budget, deadline)

        return best_x, best_f

    def _nelder_mead(self, func, call_func, x0, f0, dim, budget, deadline):
        """Nelder-Mead simplex method with adaptive parameters."""
        calls = 0
        # Initialize simplex
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        simplex = [x0.copy()]
        fvals = [f0]

        # Create other simplex points
        for i in range(dim):
            xi = x0.copy()
            xi[i] += 0.1 if x0[i] < 4.9 else -0.1
            xi = np.clip(xi, -5, 5)
            fi = call_func(xi)
            simplex.append(xi)
            fvals.append(fi)

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        max_iters = (budget - calls) // (dim + 2)
        for it in range(max_iters):
            if time.time() >= deadline:
                break

            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = call_func(x_r)

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = call_func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = call_func(x_c)
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = call_func(simplex[i])

            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if len(fvals) > 1:
                spread = fvals[-1] - fvals[0]
                if spread < 1e-10 * max(1.0, abs(fvals[0])):
                    break

        return simplex[0], fvals[0]

    def _coordinate_descent(self, func, call_func, x0, f0, dim, budget, deadline):
        """Coordinate descent with adaptive step sizes for ridge exploitation."""
        x = x0.copy()
        f = f0
        step = 1.0
        min_step = 1e-8

        for _ in range(200):  # Max 200 coordinate descent iterations
            if time.time() >= deadline:
                break
            improved = False
            for d in range(dim):
                if time.time() >= deadline:
                    break
                # Try positive and negative steps
                for direction in [1, -1]:
                    x_new = x.copy()
                    x_new[d] += direction * step
                    x_new = np.clip(x_new, -5, 5)
                    if np.allclose(x_new, x):
                        continue
                    f_new = call_func(x_new)
                    if f_new < f:
                        x = x_new
                        f = f_new
                        improved = True
                        # Adaptive step: increase if we can keep going
                        x_try = x.copy()
                        x_try[d] += direction * step * 0.5
                        x_try = np.clip(x_try, -5, 5)
                        if not np.allclose(x_try, x):
                            f_try = call_func(x_try)
                            if f_try < f:
                                x = x_try
                                f = f_try
                                step *= 1.5
                                step = min(step, 5.0)
                    else:
                        # Reduce step if no improvement
                        step *= 0.5
                        step = max(step, min_step)
                        break  # Move to next dimension

                if step <= min_step:
                    break

            if not improved and step <= min_step:
                break

        return x, f
