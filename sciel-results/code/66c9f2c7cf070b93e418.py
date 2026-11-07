"""Optimizer for BBOB-style f11 (Discus) using structure-exploiting coordinate descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        d = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return float(func(x))

        def alive():
            return calls[0] < budget and time.time() < deadline

        # ---- Phase 0: a few cheap random starts to get a low initial point ----
        rng = np.random.default_rng(12345)
        candidates = [np.zeros(d)]
        # a few random points with better coverage
        for _ in range(6):
            candidates.append(rng.uniform(-4, 4, d))
        # add points near the boundary
        for _ in range(2):
            candidates.append(rng.choice([-3.5, 3.5], d))
        best_x = None
        best_f = np.inf
        for c in candidates:
            if not alive():
                break
            c = np.clip(c, -5, 5)
            fc = wrapped(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        x = best_x.copy()
        fx = best_f

        # ---- Phase 1: per-axis 1D scans to find descent direction and scale ----
        # For each axis, sample a small set of steps in both directions to locate
        # a bracket where f decreases. This handles the rotated/anisotropic quadratic.
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
        for _ in range(4):  # a few passes; each pass refines
            for axis in range(d):
                if not alive():
                    break
                e = np.zeros(d)
                e[axis] = 1.0
                found = False
                for s in step_sizes:
                    if not alive():
                        break
                    # try both directions
                    for sign in (1.0, -1.0):
                        cand = np.clip(x + sign * s * e, -5, 5)
                        fc = wrapped(cand)
                        if fc < fx - 1e-12:
                            # descent found in this direction; record direction
                            x = cand
                            fx = fc
                            found = True
                            break
                    if found:
                        break
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # ---- Phase 2: coordinate descent with golden-section line search ----
        # The function is a positive-definite quadratic; CD with line search converges.
        phi = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio

        def line_search(axis, x0, fx0, init_step):
            # 1D minimization over t in x0 + t*e_axis, golden section
            e = np.zeros(d)
            e[axis] = 1.0

            def g(t):
                pt = np.clip(x0 + t * e, -5, 5)
                return wrapped(pt)

            # determine bracket by expanding from 0
            # first check t=0
            g0 = fx0
            t_best = 0.0
            f_best = g0
            # find a descent direction on this axis
            t_lo = -init_step
            t_hi = init_step
            # evaluate a coarse grid to bracket minimum
            grid = np.linspace(-init_step, init_step, 11)
            # find best grid point
            best_t = 0.0
            best_g = g0
            for t in grid:
                if not alive():
                    break
                gt = g(t)
                if gt < best_g:
                    best_g = gt
                    best_t = t
            if best_t == 0.0:
                # no improvement on this axis at this scale
                return x0, fx0, 0.0
            # now golden-section around best_t within [t_lo,t_hi]
            # Use a tighter bracket based on the grid result
            if best_t > 0:
                a = max(0.0, best_t - init_step * 0.5)
                b = min(init_step, best_t + init_step * 0.5)
            else:
                a = max(-init_step, best_t - init_step * 0.5)
                b = min(0.0, best_t + init_step * 0.5)
            # ensure a < b
            if a >= b:
                a, b = -init_step, init_step
            # golden section iterations - more for finer scales
            n_iters = 18 if init_step < 0.1 else 14
            for _ in range(n_iters):
                if not alive():
                    break
                m1 = a + phi * (b - a)
                m2 = b - phi * (b - a)
                f1 = g(m1)
                f2 = g(m2)
                if f1 < f2:
                    b = m2
                else:
                    a = m1
            t_mid = 0.5 * (a + b)
            fm = g(t_mid)
            if fm < f_best:
                f_best = fm
                best_t = t_mid
            pt = np.clip(x0 + best_t * e, -5, 5)
            return pt, f_best, best_t

        # iterate coordinate descent with adaptive step sizes
        step_scales = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        for scale_idx, scale in enumerate(step_scales):
            if not alive():
                break
            improved = False
            for axis in range(d):
                if not alive():
                    break
                x, fx, step = line_search(axis, x, fx, scale)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                if abs(step) > scale * 1e-6:
                    improved = True
            if not improved and scale_idx > 2:
                # No improvement at this scale and we've gone below 0.25
                # Try a few more passes at this scale before moving to finer
                for _ in range(2):
                    if not alive():
                        break
                    xi = False
                    for axis in range(d):
                        if not alive():
                            break
                        x, fx, step = line_search(axis, x, fx, scale)
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        if abs(step) > scale * 1e-7:
                            xi = True
                    if not xi:
                        break
                if not xi:
                    break
        return best_x, best_f
# EVOLVE-BLOCK-END
