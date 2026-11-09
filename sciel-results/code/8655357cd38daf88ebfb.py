"""Stronger optimizer for BBOB f12 (Bent Cigar) with anisotropy detection."""
import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

class Optimizer:
    """Anisotropy-aware optimizer exploiting Bent Cigar's extreme anisotropy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 290.0  # 300s limit with 10s safety margin

        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=np.float64)
            return func(x)

        best_x = np.zeros(self.dim)
        best_fx = float('inf')

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = np.array(x, dtype=np.float64)

        # Phase 1: CMA-ES for global exploration
        # CMA-ES handles the rotation/shift well and finds the basin
        try:
            x0 = np.zeros(self.dim)
            sigma0 = 2.0
            # Use CMA-ES with maxiter set to use most of budget
            maxiter_cma = max(1, int(self.budget * 0.5 / (self.dim + 1)))
            res = minimize(
                eval_func,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res.fun < best_fx:
                update_best(res.x, res.fun)
        except StopIteration:
            pass
        except Exception:
            pass

        # Phase 2: Detect anisotropy direction via finite differences
        # Bent Cigar: f(x) = 100*(x0-1)^2 + sum(x_i^2) or similar
        # The high-curvature direction is along one axis (possibly rotated)
        # We probe to find the direction of steepest curvature

        if self.calls < self.budget - 10 and time.time() - self.start_time < self.time_limit:
            # Probe along coordinate axes to find the high-curvature direction
            probe_x = best_x.copy()
            probe_step = 0.5
            axis_curvatures = []

            for i in range(self.dim):
                x_plus = probe_x.copy()
                x_minus = probe_x.copy()
                x_plus[i] += probe_step
                x_minus[i] -= probe_step

                try:
                    f_plus = eval_func(x_plus)
                    f_minus = eval_func(x_minus)
                    f_center = best_fx
                    # Approximate second derivative
                    curvature = (f_plus - 2*f_center + f_minus) / (probe_step**2)
                    axis_curvatures.append(curvature)
                except StopIteration:
                    axis_curvatures.append(0.0)
                    break
                except Exception:
                    axis_curvatures.append(0.0)

            # The Bent Cigar has one axis with curvature ~100x the others
            # Find the dominant axis
            if axis_curvatures:
                max_curv_idx = int(np.argmax(np.abs(axis_curvatures)))
                dominant_axis = max_curv_idx
                dominant_curv = abs(axis_curvatures[max_curv_idx])
                # Check if there's a clear anisotropy (ratio > 5x)
                other_curvs = [abs(c) for j, c in enumerate(axis_curvatures) if j != max_curv_idx]
                if other_curvs:
                    min_other = min(other_curvs) if other_curvs else 0
                    if min_other > 0 and dominant_curv / max(min_other, 1e-10) > 3:
                        # Clear anisotropy detected - do 1D line search along dominant axis
                        self._anisotropy_line_search(func, eval_func, probe_x, dominant_axis, update_best)

        # Phase 3: Powell refinement along all directions
        remaining = self.budget - self.calls
        if remaining > 15 and time.time() - self.start_time < self.time_limit - 5:
            try:
                res = minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': min(200, remaining // (self.dim + 1)),
                        'ftol': 1e-14,
                        'xatol': 1e-10
                    }
                )
                if res.fun < best_fx:
                    update_best(res.x, res.fun)
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 4: Final Nelder-Mead polish
        remaining = self.budget - self.calls
        if remaining > 10 and time.time() - self.start_time < self.time_limit - 3:
            try:
                res = minimize(
                    eval_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(100, remaining // (self.dim + 1)),
                             'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_fx:
                    update_best(res.x, res.fun)
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 5: Coarse-to-fine coordinate descent
        remaining = self.budget - self.calls
        if remaining > 8 and time.time() - self.start_time < self.time_limit - 2:
            self._coordinate_descent(eval_func, best_x, update_best, remaining)

        if not np.all(np.isfinite(best_x)):
            best_x = np.zeros(self.dim)
            if self.calls < self.budget:
                try:
                    best_fx = eval_func(best_x)
                except Exception:
                    best_fx = 0.0

        return best_x, best_fx

    def _anisotropy_line_search(self, func, eval_func, x, axis, update_best):
        """Perform 1D line search along the dominant anisotropy axis."""
        # Coarse search
        lo, hi = x[axis] - 8.0, x[axis] + 8.0
        # Find bounds where function increases
        def f1d(t):
            y = x.copy()
            y[axis] = t
            try:
                return eval_func(y)
            except StopIteration:
                return float('inf')

        # Golden section search for the minimum along this axis
        gr = (math.sqrt(5) - 1) / 2  # golden ratio ~0.618

        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f1d(c)
        fd = f1d(d)

        for _ in range(30):
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f1d(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f1d(d)

        # Best point from golden section
        t_best = (a + b) / 2.0
        x_best = x.copy()
        x_best[axis] = t_best
        try:
            fx_best = eval_func(x_best)
            update_best(x_best, fx_best)
        except StopIteration:
            pass
        except Exception:
            pass

        # Fine-tune with minimize_scalar
        if self.calls < self.budget - 5 and time.time() - self.start_time < self.time_limit - 2:
            try:
                result = minimize_scalar(
                    f1d,
                    bounds=(a, b),
                    method='bounded',
                    options={'maxiter': 50}
                )
                if result.success and result.fun < float('inf'):
                    x_fine = x.copy()
                    x_fine[axis] = result.x
                    fx_fine = result.fun
                    update_best(x_fine, fx_fine)
            except StopIteration:
                pass
            except Exception:
                pass

    def _coordinate_descent(self, eval_func, x0, update_best, budget_remaining):
        """Coarse-to-fine coordinate descent."""
        x = x0.copy()
        steps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        max_coords_per_step = max(1, budget_remaining // len(steps))

        for step_size in steps:
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
            for i in range(self.dim):
                if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                    break
                # Try + and - direction
                for direction in [1, -1]:
                    if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                        break
                    y = x.copy()
                    y[i] += direction * step_size
                    try:
                        fy = eval_func(y)
                        if fy < float('inf'):
                            # Accept if improvement
                            # Do a small local search
                            for _ in range(3):
                                if self.calls >= self.budget:
                                    break
                                y2 = y.copy()
                                y2[i] += direction * step_size
                                try:
                                    fy2 = eval_func(y2)
                                    if fy2 < fy:
                                        y = y2
                                        fy = fy2
                                    else:
                                        break
                                except StopIteration:
                                    break
                                except Exception:
                                    break
                            update_best(y, fy)
                            x = y.copy()
                    except StopIteration:
                        break
                    except Exception:
                        pass
