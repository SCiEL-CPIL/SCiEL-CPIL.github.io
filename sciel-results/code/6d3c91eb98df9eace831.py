"""Black-box continuous optimizer on the BBOB-style Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: quadratic-model Newton jumps + Powell polish + line-search ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _fd_grad(self, func, x, calls, h):
        # central finite differences, 2*dim calls
        g = np.empty(self.dim)
        for i in range(self.dim):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g[i] = (func(np.clip(xp, -5, 5)) - func(np.clip(xm, -5, 5))) / (2 * h)
        calls[0] += 2 * self.dim
        return g

    def _quadratic_step(self, func, x, fx, calls, h):
        # Build a local quadratic model from gradient + finite-difference Hessian.
        g = self._fd_grad(func, x, calls, h)
        # Hessian via forward differences on the gradient (dim more gradient evals = 2*dim*dim calls)
        # That is expensive; instead use Gauss-Newton style: approximate Hessian as J^T J
        # where J is the Jacobian of the gradient (same cost). We use a diagonal + rank-1 model.
        # Simpler robust model: use gradient direction with a curvature estimate.
        # Estimate curvature along the gradient direction via a 3-point probe.
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            return None, g
        d = g / ng
        # probe quadratic along d
        a = 0.05
        fa = func(np.clip(x + a * d, -5, 5)); calls[0] += 1
        fb = func(np.clip(x - a * d, -5, 5)); calls[0] += 1
        # second difference ~ 2*a^2 * (d^T H d)
        curv = (fa + fb - 2 * fx) / (2 * a * a)
        if curv < 1e-10:
            curv = 1e-10
        # step length for the quadratic minimizer along d: alpha = -g_d / curv
        gd = np.dot(g, d)  # = -ng
        alpha = -gd / curv
        # cap the step
        alpha = np.clip(alpha, 0.0, 5.0)
        return d * alpha, g

    def _line_search(self, func, x, fx, d, calls, max_step=5.0):
        # Backtracking line search along direction d
        alpha = 1.0
        for _ in range(20):
            xn = np.clip(x + alpha * d, -5, 5)
            fxn = func(xn); calls[0] += 1
            if fxn < fx:
                return xn, fxn
            alpha *= 0.5
        return x, fx

    def _powell(self, func, x, fx, calls, budget_left):
        # Powell's conjugate-direction method.
        n = self.dim
        dirs = np.eye(n)
        fx_best = fx
        x_best = x.copy()
        while calls[0] < self.budget - 2 and budget_left > 0:
            x_old = x.copy()
            f_old = fx
            for i in range(n):
                # line search along dirs[i]
                d = dirs[i]
                # find optimal alpha by ternary search in [-1, 1] scaled
                lo, hi = -1.0, 1.0
                f_lo = func(np.clip(x + lo * d, -5, 5)); calls[0] += 1
                f_hi = func(np.clip(x + hi * d, -5, 5)); calls[0] += 1
                for _ in range(6):
                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0
                    f_m1 = func(np.clip(x + m1 * d, -5, 5)); calls[0] += 1
                    f_m2 = func(np.clip(x + m2 * d, -5, 5)); calls[0] += 1
                    if f_m1 < f_m2:
                        hi = m2; f_hi = f_m2
                    else:
                        lo = m1; f_lo = f_m1
                    if calls[0] >= self.budget - 1:
                        break
                a = (lo + hi) / 2.0
                x_new = np.clip(x + a * d, -5, 5)
                f_new = func(x_new); calls[0] += 1
                if f_new < fx:
                    x, fx = x_new, f_new
                else:
                    x = x_old.copy()
                    fx = f_old
                if calls[0] >= self.budget - 1:
                    break
            if calls[0] >= self.budget - 1:
                break
            # conjugate direction
            d_conj = x - x_old
            if np.linalg.norm(d_conj) < 1e-12:
                break
            x_start = x_old.copy()
            f_start = f_old
            # line search along d_conj
            lo, hi = -1.0, 1.0
            f_lo = func(np.clip(x_start + lo * d_conj, -5, 5)); calls[0] += 1
            f_hi = func(np.clip(x_start + hi * d_conj, -5, 5)); calls[0] += 1
            for _ in range(6):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                f_m1 = func(np.clip(x_start + m1 * d_conj, -5, 5)); calls[0] += 1
                f_m2 = func(np.clip(x_start + m2 * d_conj, -5, 5)); calls[0] += 1
                if f_m1 < f_m2:
                    hi = m2; f_hi = f_m2
                else:
                    lo = m1; f_lo = f_m1
                if calls[0] >= self.budget - 1:
                    break
            a = (lo + hi) / 2.0
            x_conj = np.clip(x_start + a * d_conj, -5, 5)
            f_conj = func(x_conj); calls[0] += 1
            if f_conj < fx:
                x, fx = x_conj, f_conj
            # replace the worst direction
            f_delta = f_old - f
            # find direction with largest f increase
            worst = 0
            worst_val = -np.inf
            for i in range(n):
                pass
            # simple: replace last direction
            dirs[-1] = d_conj
            if np.linalg.norm(x - x_best) < 1e-9 and f_conj >= fx_best:
                break
            if fx < fx_best:
                fx_best = fx
                x_best = x.copy()
        return x_best, fx_best

    def __call__(self, func):
        calls = [0]
        start = time.time()
        deadline = start + 280.0  # safety margin before 300s

        def remaining():
            return time.time() < deadline

        # Phase 1: random start, then gradient-based quadratic jumps
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x); calls[0] = 1

        # Try a few quadratic-model Newton-like steps for big early gains
        h = 0.01
        for it in range(30):
            if calls[0] >= self.budget - 5 or not remaining():
                break
            step, g = self._quadratic_step(func, x, fx, calls, h)
            if step is None:
                break
            xn, fxn = self._line_search(func, x, fx, step, calls, max_step=5.0)
            if fxn < fx:
                x, fx = xn, fxn
                h *= 0.7
            else:
                h *= 0.5
                if h < 1e-4:
                    break

        # Phase 2: Powell's method to polish the valley
        if calls[0] < self.budget - 2 and remaining():
            x, fx = self._powell(func, x, fx, calls, self.budget - calls[0])

        # Phase 3: local CMA-ES-like line search to squeeze the last bits
        sigma = 0.1
        while calls[0] < self.budget - 1 and remaining():
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y); calls[0] += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.85
                if sigma < 1e-6:
                    break

        return x, fx
# EVOLVE-BLOCK-END
