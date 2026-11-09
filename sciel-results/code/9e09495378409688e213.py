"""Two-basin basin-fitting with direct basin jumps + Nelder-Mead / coordinate-descent polish."""
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
            # Map to [-5, 5]
            p = [2 * 10 * v - 5 for v in p]
            pts.append(np.array(p))
        return pts

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        calls = [0]
        dim = self.dim
        budget = self.budget

        def f(x):
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        best_x = None
        best_f = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_x = np.array(x, dtype=float)
                best_f = fx

        # --- Phase 1: Coarse exploration via Halton + strategic points ---
        halton_pts = self._halton_sequence(20, dim)
        strategic = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0][:dim]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0][:dim]),
            np.array([2.0, 0.0, -2.0, 0.0, 2.0][:dim]),
            np.array([0.0, 2.0, 0.0, -2.0, 0.0][:dim]),
            np.array([3.0, -3.0, 3.0, -3.0, 3.0][:dim]),
            np.array([-3.0, 3.0, -3.0, 3.0, -3.0][:dim]),
            np.array([1.5, 1.5, -1.5, -1.5, 1.5][:dim]),
            np.array([-1.5, -1.5, 1.5, 1.5, -1.5][:dim]),
        ]

        all_starts = []
        for p in halton_pts:
            all_starts.append(np.clip(p, -5, 5))
        for s in strategic:
            all_starts.append(np.clip(s, -5, 5))

        # Evaluate all starts
        evaluated = []
        for x0 in all_starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            fx = f(x0)
            update_best(x0, fx)
            if np.isfinite(fx):
                evaluated.append((x0.copy(), fx))

        # Sort by fitness
        evaluated.sort(key=lambda t: t[1])

        # --- Phase 2: Basin-fitting via local quadratic approximation ---
        # For the best few points, fit a local quadratic to estimate basin center
        # and jump to it. This is the key structural idea: use the shape of the
        # function around a point to predict where the basin minimum is.

        def fit_quadratic_jump(x0, fx0, n_samples=12):
            """Fit a local quadratic around x0 and return the predicted minimum."""
            if calls[0] + n_samples > budget:
                return None
            if time.time() > deadline:
                return None

            # Sample points around x0 in a small ball
            radius = 0.5
            # Use deterministic offsets: coordinate directions and some diagonals
            offsets = []
            for d in range(dim):
                for sign in [1, -1]:
                    off = np.zeros(dim)
                    off[d] = sign * radius
                    offsets.append(off)
            # Add a few diagonal offsets
            for k in range(min(4, dim)):
                off = np.zeros(dim)
                off[k] = radius * 0.7
                if k + 1 < dim:
                    off[k + 1] = radius * 0.7
                offsets.append(off)

            # Collect samples
            X = []
            Y = []
            for off in offsets:
                if calls[0] >= budget or time.time() > deadline:
                    break
                xp = np.clip(x0 + off, -5, 5)
                if np.allclose(xp, x0, atol=1e-10):
                    continue
                fpx = f(xp)
                if np.isfinite(fpx):
                    X.append(xp)
                    Y.append(fpx)

            if len(X) < dim + 1:
                return None

            X = np.array(X)
            Y = np.array(Y)

            # Fit quadratic: f(x) ≈ c + b^T(x - x0) + (x - x0)^T A (x - x0)
            # We only need the linear part and the Hessian approximation.
            # Use least squares to fit: Y ≈ c + B (X - x0)
            # where B is a (dim+1) vector: [b_0, ..., b_{d-1}]
            # Actually, let's fit a full quadratic in the local coordinates.
            # Local coords: u = X - x0, shape (n, dim)
            u = X - x0
            # Build feature matrix: [1, u_1, ..., u_d, u_1^2, ..., u_d^2, u_1*u_2, ...]
            # For simplicity, fit linear + quadratic diagonal terms only (no cross terms)
            # This is a paraboloid approximation.
            n_feat = 1 + dim + dim  # intercept + linear + quadratic diagonal
            F = np.ones((len(u), n_feat))
            F[:, 1:1 + dim] = u
            for j in range(dim):
                F[:, 1 + dim + j] = u[:, j] ** 2

            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(F, Y, rcond=None)
            except np.linalg.LinAlgError:
                return None

            # coeffs[0] = intercept, coeffs[1:1+dim] = linear, coeffs[1+dim:] = quadratic
            # The minimum of c + b^T u + sum_j a_j u_j^2 is at u* where:
            # b_j + 2 a_j u*_j = 0 => u*_j = -b_j / (2 a_j) if a_j != 0
            u_star = np.zeros(dim)
            valid = True
            for j in range(dim):
                a_j = coeffs[1 + dim + j]
                b_j = coeffs[1 + j]
                if abs(a_j) < 1e-12:
                    # Linear term only; move in direction of -b_j
                    u_star[j] = -b_j * 0.1  # small step
                else:
                    u_star[j] = -b_j / (2.0 * a_j)

            # Clamp the jump to be reasonable
            u_star = np.clip(u_star, -3.0, 3.0)
            x_new = np.clip(x0 + u_star, -5, 5)
            return x_new

        # Try basin jumps from the top 3 best evaluated points
        for i, (x0, fx0) in enumerate(evaluated[:3]):
            if calls[0] >= budget or time.time() > deadline:
                break
            x_jump = fit_quadratic_jump(x0, fx0)
            if x_jump is not None:
                fx_jump = f(x_jump)
                update_best(x_jump, fx_jump)
                if np.isfinite(fx_jump):
                    evaluated.append((x_jump.copy(), fx_jump))

        # Re-sort
        evaluated.sort(key=lambda t: t[1])

        # --- Phase 3: Nelder-Mead refinement from the best point ---
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            remaining = budget - calls[0]
            maxfev_nm = min(remaining // 3, 150)
            if maxfev_nm >= 10:
                try:
                    res = minimize(f, best_x, method='Nelder-Mead',
                                   options={'maxiter': 200, 'maxfev': maxfev_nm,
                                            'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        # --- Phase 4: Try Nelder-Mead from second-best if it's far from best ---
        if calls[0] < budget and time.time() < deadline:
            for x0, fx0 in evaluated[1:4]:
                if calls[0] >= budget or time.time() > deadline:
                    break
                if best_x is not None and np.linalg.norm(x0 - best_x) < 0.5:
                    continue
                remaining = budget - calls[0]
                maxfev_nm = min(remaining // 4, 80)
                if maxfev_nm < 10:
                    break
                try:
                    res = minimize(f, x0, method='Nelder-Mead',
                                   options={'maxiter': 100, 'maxfev': maxfev_nm,
                                            'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        # --- Phase 5: Coordinate descent polish with geometric step decay ---
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            current_x = best_x.copy()
            current_f = best_f

            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005,
                          0.002, 0.001, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5,
                          5e-6, 2e-6, 1e-6]

            for sigma in step_sizes:
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
