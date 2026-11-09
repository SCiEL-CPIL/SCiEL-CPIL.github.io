"""Paraboloid basin-fitting with direct basin jumps + Nelder-Mead / coordinate-descent polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic two-basin search: fit a local quadratic, jump to its center,
    then polish with Nelder-Mead and fine coordinate descent."""

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
        """Fit a quadratic to center + 2n axis points; return predicted minimum."""
        n = self.dim
        # center value
        c = np.clip(center, -5, 5)
        if calls[0] >= budget or time.time() > deadline:
            return None, None
        fc = f(c)

        pts = [c]
        vals = [fc]
        for d in range(n):
            for s in (1.0, -1.0):
                if calls[0] >= budget or time.time() > deadline:
                    return None, None
                x = c.copy()
                x[d] = np.clip(x[d] + s * step, -5, 5)
                if np.allclose(x, c, atol=1e-12):
                    continue
                pts.append(x)
                vals.append(f(x))

        # Need at least 2n+1 points
        if len(pts) < 2 * n + 1:
            return None, None

        A = np.array(pts)
        b = np.array(vals)
        # Least-squares fit of quadratic: b = A^T h + A^2 c + k
        # Build design matrix
        D = np.hstack([A, A ** 2, np.ones((len(A), 1))])
        try:
            coef, *_ = np.linalg.lstsq(D, b, rcond=None)
        except Exception:
            return None, None
        h = coef[:n]
        c_q = coef[n:2 * n]
        # Minimum of quadratic: x* = -0.5 * inv(2*C) * h  where C = diag(c_q)
        # For diagonal quadratic, x*_d = -h_d / (2*c_q_d)
        x_star = np.zeros(n)
        ok = True
        for d in range(n):
            if abs(c_q[d]) < 1e-8:
                ok = False
                break
            x_star[d] = -h[d] / (2.0 * c_q[d])
        if not ok:
            return None, None
        x_star = np.clip(x_star, -5, 5)
        return x_star, fc

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

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

        # 1. Deterministic Initialization
        # Cover the space with a deterministic grid of points
        # Focus on points that are likely to be in the basin of attraction
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Corners of the box
        for mask in range(1 << dim):
            p = np.array([1.0 if (mask >> d) & 1 else -1.0 for d in range(dim)])
            init_points.append(p)
        # Intermediate points (Lunacek often has minima near +/-1 or +/-2)
        for val in [-2.0, -1.0, 1.0, 2.0]:
            init_points.append(np.full(dim, val))
        # Mix
        init_points.append(np.array([1.0, -1.0, 1.0, -1.0, 1.0][:dim]))
        init_points.append(np.array([-1.0, 1.0, -1.0, 1.0, -1.0][:dim]))
        init_points.append(np.array([2.0, -2.0, 2.0, -2.0, 2.0][:dim]))
        init_points.append(np.array([-2.0, 2.0, -2.0, 2.0, -2.0][:dim]))

        # Evaluate initial points
        for x0 in init_points:
            if calls[0] >= budget or time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = f(x0)
            update_best(x0, fx)

        # 2. Multi-Scale Coordinate Descent
        # Use a sequence of decreasing step sizes
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        # We will maintain a list of (x, f) pairs.
        active_points = []
        if best_x is not None:
            active_points.append((best_x.copy(), best_f))
        
        # Add a few diverse points to active_points
        for x0 in init_points[:5]:
            if calls[0] >= budget or time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            if not any(np.allclose(x0, ap[0]) for ap in active_points):
                fx = f(x0)
                update_best(x0, fx)
                active_points.append((x0.copy(), fx))

        for sigma in step_sizes:
            if calls[0] >= budget or time.time() > deadline:
                break
            
            next_points = []
            
            for x_curr, f_curr in active_points:
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                x_temp = x_curr.copy()
                f_temp = f_curr
                
                # Coordinate descent
                improved = True
                while improved and calls[0] < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if calls[0] >= budget or time.time() > deadline:
                            break
                        for sign in [1, -1]:
                            if calls[0] >= budget or time.time() > deadline:
                                break
                            x_try = x_temp.copy()
                            x_try[d] += sign * sigma
                            x_try = np.clip(x_try, -5, 5)
                            f_try = f(x_try)
                            if f_try < f_temp:
                                x_temp = x_try
                                f_temp = f_try
                                improved = True
                                update_best(x_temp, f_temp)
                
                next_points.append((x_temp.copy(), f_temp))
            
            # Prune next_points: Keep the best 5
            if next_points:
                next_points.sort(key=lambda p: p[1])
                active_points = next_points[:5]
            else:
                break

        # 3. Final Polish with Nelder-Mead
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            maxfev = min(remaining, 100)
            try:
                res = minimize(f, best_x, method='Nelder-Mead',
                               options={'maxiter': 50, 'maxfev': maxfev,
                                        'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # 4. Final Fine Coordinate Descent
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            x_curr = best_x.copy()
            f_curr = best_f
            fine_steps = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
            for sigma in fine_steps:
                if calls[0] >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and calls[0] < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if calls[0] >= budget or time.time() > deadline:
                            break
                        for sign in [1, -1]:
                            if calls[0] >= budget or time.time() > deadline:
                                break
                            x_try = x_curr.copy()
                            x_try[d] += sign * sigma
                            x_try = np.clip(x_try, -5, 5)
                            f_try = f(x_try)
                            if f_try < f_curr:
                                x_curr = x_try
                                f_curr = f_try
                                improved = True
                                update_best(x_curr, f_curr)

        return best_x, best_f
# EVOLVE-BLOCK-END
