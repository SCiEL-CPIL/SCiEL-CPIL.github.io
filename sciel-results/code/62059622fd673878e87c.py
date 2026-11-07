"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Sobol global screen + Nelder-Mead refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 290.0   # safety margin before the 300 s wall-clock limit

        # ---------- evaluation tracker ----------
        self._count = 0
        self._best = None
        self._bestx = None

        def ev(x):
            self._count += 1
            v = float(func(np.asarray(x, dtype=float)))
            if self._best is None or v < self._best:
                self._best = v
                self._bestx = np.asarray(x, dtype=float).copy()
            return v

        # ---------- 1) global screen with a Sobol sequence ----------
        # Deterministic, cheap, spreads the whole [-5,5]^d box to catch a basin early.
        n_screen = 240
        pts = self._sobol(n_screen, d)
        for p in pts:
            ev(p)
        if self._count >= budget:
            return self._bestx, self._best

        # ---------- 2) Nelder-Mead refinement on the best candidate ----------
        x0 = self._bestx.copy()
        self._nm(x0, ev, deadline)
        if self._count >= budget:
            return self._bestx, self._best

        # ---------- 3) spend remaining budget on the incumbent ----------
        # A bounded (1+1)-ES-style local walk to squeeze out more improvement.
        rng = np.random.default_rng(0)
        sigma = 0.15
        while self._count < budget:
            if time.time() > deadline:
                break
            y = np.clip(self._bestx + sigma * rng.standard_normal(d), -5.0, 5.0)
            fy = ev(y)
            if fy < self._best:
                sigma *= 1.4
            else:
                sigma *= 0.85
                if sigma < 1e-3:
                    sigma = 1e-3
        return self._bestx, self._best

    # ---------- Sobol quasi-random points in [-5,5]^d ----------
    @staticmethod
    def _sobol(n, d):
        # VDC (Van der Corput) base-2 sequence per dimension: cheap, low-discrepancy.
        pts = np.empty((n, d))
        for j in range(d):
            for i in range(n):
                x = 0.0
                den = 1.0
                m = i + 1
                while m > 0:
                    den *= 2.0
                    x += (m & 1) / den
                    m >>= 1
                pts[i, j] = x
        return pts * 10.0 - 5.0

    # ---------- Nelder-Mead simplex (bounded, time-guarded) ----------
    def _nm(self, x0, ev, deadline):
        d = self.dim
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        # initial simplex around x0
        xk = [x0.copy() for _ in range(d + 1)]
        f = [None] * (d + 1)
        for i in range(1, d + 1):
            xk[i] = x0.copy()
            xk[i][i - 1] += 0.5
            xk[i] = np.clip(xk[i], -5.0, 5.0)
        for i in range(d + 1):
            f[i] = ev(xk[i])
            if self._count >= self.budget:
                return

        iters = 0
        max_iters = 1200
        while iters < max_iters:
            iters += 1
            if self._count >= self.budget or time.time() > deadline:
                break

            # order by objective
            order = np.argsort(f)
            xk = [xk[i].copy() for i in order]
            f = [f[i] for i in order]
            xw, fw = xk[-1], f[-1]          # worst
            xg, fg = xk[0], f[0]            # best

            # centroid of all but worst
            xc = np.mean(xk[:-1], axis=0)

            # reflection
            xr = np.clip(xc + alpha * (xc - xw), -5.0, 5.0)
            fr = ev(xr)
            if self._count >= self.budget:
                break

            if fr < f[0]:
                # expansion
                xe = np.clip(xc + gamma * (xc - xw), -5.0, 5.0)
                fe = ev(xe)
                if self._count >= self.budget:
                    break
                if fe < fr:
                    xk[-1], f[-1] = xe, fe
                else:
                    xk[-1], f[-1] = xr, fr
            elif fr < f[-2]:
                xk[-1], f[-1] = xr, fr
            else:
                # contraction
                if fr < fw:
                    # outside contraction
                    xc2 = np.clip(xc + 0.5 * (xr - xc), -5.0, 5.0)
                else:
                    # inside contraction
                    xc2 = np.clip(xc - 0.5 * (xr - xc), -5.0, 5.0)
                fc = ev(xc2)
                if self._count >= self.budget:
                    break
                if fc <= min(fr, fw):
                    xk[-1], f[-1] = xc2, fc
                else:
                    # shrink toward best
                    for i in range(1, d + 1):
                        xk[i] = np.clip(xg + sigma * (xk[i] - xg), -5.0, 5.0)
                        f[i] = ev(xk[i])
                        if self._count >= self.budget:
                            break
                    if self._count >= self.budget:
                        break
                    # stop if simplex collapsed
                    if np.max(f) - np.min(f) < 1e-12:
                        break
        return
# EVOLVE-BLOCK-END
