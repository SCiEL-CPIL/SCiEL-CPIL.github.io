"""Rewritten optimizer: multi-start Nelder-Mead with time guard for Rastrigin f3."""
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with deterministic simplex vertices and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, max_iter, call_count):
        """Run Nelder-Mead from x0, returning best point and best value.
        call_count is a mutable list [count] to track evaluations."""
        d = self.dim
        # Build initial simplex
        simplex = [x0.copy()]
        for i in range(d):
            v = x0.copy()
            step = 0.25 if x0[i] + 0.25 <= 5 else -0.25
            v[i] += step
            simplex.append(np.clip(v, -5, 5))

        vals = np.empty(d + 1)
        for i, v in enumerate(simplex):
            vals[i] = func(v)
            call_count[0] += 1

        for _ in range(max_iter):
            if call_count[0] >= self.budget - 5:
                break

            # Sort by function value
            order = np.argsort(vals)
            simplex = [simplex[i] for i in order]
            vals = vals[order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            call_count[0] += 1

            if f_ref < vals[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                call_count[0] += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    vals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    vals[-1] = f_ref
            elif f_ref < vals[-2]:
                simplex[-1] = x_ref
                vals[-1] = f_ref
            else:
                # Contraction
                rho = 0.5
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                call_count[0] += 1
                if f_con < vals[-1]:
                    simplex[-1] = x_con
                    vals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        vals[i] = func(simplex[i])
                        call_count[0] += 1
                    if call_count[0] >= self.budget - 5:
                        break

        best_idx = np.argmin(vals)
        return simplex[best_idx], vals[best_idx]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s
        d = self.dim
        call_count = [0]

        # Generate 10 deterministic starting points at vertices of a regular 5-simplex
        # inscribed in [-5, 5]^5, plus center
        # Use a structured set: center, and points offset along each axis
        starts = []
        # Center
        starts.append(np.zeros(d))
        # Axis-aligned points
        for i in range(d):
            for sign in [1, -1]:
                p = np.zeros(d)
                p[i] = sign * 4.0
                starts.append(p)
        # Some additional diverse points
        starts.append(np.array([3.0, -3.0, 2.0, -2.0, 3.0]))
        starts.append(np.array([-2.0, 3.0, -3.0, 2.0, -2.0]))
        starts.append(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        starts.append(np.array([-1.0, -1.0, -1.0, -1.0, -1.0]))

        # Evaluate all starts first to rank them
        start_vals = []
        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            v = func(s_clipped)
            call_count[0] += 1
            start_vals.append((v, s_clipped))
            if call_count[0] >= self.budget:
                break

        # Sort by value, best first
        start_vals.sort(key=lambda x: x[0])

        best_x = start_vals[0][1]
        best_f = start_vals[0][0]

        # Run Nelder-Mead from each start, allocating remaining budget
        remaining = self.budget - call_count[0]
        n_starts = len(start_vals)
        if n_starts == 0:
            return best_x, best_f

        # Allocate iterations: give more to better starts
        # Use a simple scheme: top start gets 40%, next 25%, next 20%, rest 15%
        if n_starts == 1:
            iters = [max(1, remaining // 2)]
        else:
            fractions = [0.4, 0.25, 0.2, 0.15]
            while len(fractions) < n_starts:
                fractions.append(0.15)
            fractions = fractions[:n_starts]
            total_frac = sum(fractions)
            fractions = [f / total_frac for f in fractions]
            iters = [max(1, int(remaining * f)) for f in fractions]

        for i, (f_val, x_start) in enumerate(start_vals):
            if time.time() > deadline:
                break
            if call_count[0] >= self.budget - 5:
                break

            max_iter = iters[i]
            x_res, f_res = self._nelder_mead(func, x_start, max_iter, call_count)

            if f_res < best_f:
                best_f = f_res
                best_x = x_res

            # Check time
            if time.time() > deadline:
                break

        return best_x, best_f
