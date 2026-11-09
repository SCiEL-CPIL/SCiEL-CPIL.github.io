"""Two-basin quadratic mixture model for Lunacek bi-Rastrigin + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Exploits the known two-basin quadratic structure of Lunacek bi-Rastrigin.
    Uses a fixed set of evaluations to solve for basin centers via linear algebra,
    then polishes with Nelder-Mead and fine coordinate descent."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def _halton_sequence(self, n, dim):
        pts = []
        for i in range(1, n + 1):
            p = [self._halton(i, b) for b in range(2, dim + 2)]
            p = [10.0 * q - 5.0 for q in p]
            pts.append(np.array(p, dtype=float))
        return pts

    def _fit_paraboloid(self, center, step, f, calls, budget, deadline):
        """Fit a diagonal quadratic to center + 2n axis points; return predicted minimum."""
        n = self.dim
        c = np.clip(center, -5, 5)
        if calls[0] >= budget or time.time() > deadline:
            return None
        fc = f(c)

        pts = [c]
        vals = [fc]
        for d in range(n):
            for s in (1.0, -1.0):
                if calls[0] >= budget or time.time() > deadline:
                    return None
                x = c.copy()
                x[d] = np.clip(x[d] + s * step, -5, 5)
                if np.allclose(x, c, atol=1e-12):
                    continue
                pts.append(x)
                vals.append(f(x))

        if len(pts) < 2 * n + 1:
            return None

        A = np.array(pts)
        b = np.array(vals)
        D = np.hstack([A, A ** 2, np.ones((len(A), 1))])
        try:
            coef, *_ = np.linalg.lstsq(D, b, rcond=None)
        except Exception:
            return None
        h = coef[:n]
        c_q = coef[n:2 * n]
        x_star = np.zeros(n)
        ok = True
        for d in range(n):
            if abs(c_q[d]) < 1e-8:
                ok = False
                break
            x_star[d] = -h[d] / (2.0 * c_q[d])
        if not ok:
            return None
        x_star = np.clip(x_star, -5, 5)
        return x_star

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        calls = [0]
        dim = self.dim
        budget = self.budget

        def f(x):
            if calls[0] >= budget or time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        # ---------- Phase 1: diverse exploration ----------
        halton_points = self._halton_sequence(12, dim)
        candidate_starts = list(halton_points)
        candidate_starts.append(np.zeros(dim))
        candidate_starts.append(np.full(dim, 1.0))
        candidate_starts.append(np.full(dim, -1.0))
        candidate_starts.append(np.array([1.0, -1.0, 1.0, -1.0, 1.0][:dim]))
        candidate_starts.append(np.array([-1.0, 1.0, -1.0, 1.0, -1.0][:dim]))

        evaluated_starts = []
        for x0 in candidate_starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = f(x0)
            update_best(x0, fx)
            evaluated_starts.append((x0, fx))

        # ---------- Phase 2: paraboloid basin jumps from top starts ----------
        evaluated_starts.sort(key=lambda t: t[1])
        jump_targets = []
        for x0, fx0 in evaluated_starts[:5]:
            if calls[0] >= budget or time.time() > deadline:
                break
            for step in (1.0, 2.0, 3.0):
                x_star = self._fit_paraboloid(x0, step, f, calls, budget, deadline)
                if x_star is not None:
                    jump_targets.append(x_star)

        # Evaluate jump targets
        for xt in jump_targets:
            if calls[0] >= budget or time.time() > deadline:
                break
            xt = np.clip(xt, -5, 5)
            fx = f(xt)
            update_best(xt, fx)
            evaluated_starts.append((xt, fx))

        # ---------- Phase 3: Nelder-Mead from best and diverse points ----------
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            maxfev = min(remaining // 2, 120)
            try:
                res = minimize(f, best_x, method='Nelder-Mead',
                               options={'maxiter': 60, 'maxfev': maxfev,
                                        'xatol': 1e-9, 'fatol': 1e-9})
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Nelder-Mead from diverse points
        if calls[0] < budget and time.time() < deadline:
            evaluated_starts.sort(key=lambda t: t[1])
            for x0, fx0 in evaluated_starts[:10]:
                if calls[0] >= budget or time.time() > deadline:
                    break
                if best_x is not None and np.linalg.norm(x0 - best_x) < 0.3:
                    continue
                remaining = budget - calls[0]
                maxfev = min(remaining // 4, 60)
                if maxfev < 10:
                    break
                try:
                    res = minimize(f, x0, method='Nelder-Mead',
                                   options={'maxiter': 30, 'maxfev': maxfev,
                                            'xatol': 1e-9, 'fatol': 1e-9})
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        # ---------- Phase 4: fine coordinate-descent polish ----------
        if calls[0] < budget and time.time() < deadline and best_x is not None:
            current_x = best_x.copy()
            current_f = best_f
            fine_steps = [5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
            for sigma in fine_steps:
                if calls[0] >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and calls[0] < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if calls[0] >= budget or time.time() > deadline:
                            break
                        for sign in (1, -1):
                            if calls[0] >= budget or time.time() > deadline:
                                break
                            new_x = current_x.copy()
                            new_x[d] = np.clip(new_x[d] + sign * sigma, -5, 5)
                            if np.allclose(new_x, current_x, atol=1e-12):
                                continue
                            fx = f(new_x)
                            if fx < current_f:
                                current_x = new_x
                                current_f = fx
                                update_best(current_x, current_f)
                                improved = True

        return best_x, best_f
# EVOLVE-BLOCK-END
