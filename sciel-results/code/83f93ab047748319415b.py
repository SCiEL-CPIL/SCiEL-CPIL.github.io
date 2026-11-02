"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the classical upper bound on A(f)*A(Fourier(f)) over the three free
Hermite coefficients (degrees 0, 4, 8).  The degree-12 weight is fixed so that
g(0) = 0.  The bound is
    B(a) = max_{lambda>=0} (1 - g(lambda)) / g(lambda)
        = max_{lambda>=0} (1 / g(lambda) - 1),
so minimizing B is equivalent to maximizing the minimum of g over lambda>=0.
We minimize the smooth surrogate  -softmin g  (log-sum-exp) with an analytic
gradient, add a change-of-variables and a basin-hopping global pass to escape
local optima, then a high-precision polish on the exact max objective.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 20.0          # safety margin (seconds) before the hard kill
_START = None


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    x2 = x * x
    h0 = np.ones_like(x)
    h4 = 16.0 * x2 * x2 - 48.0 * x2 + 12.0
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x8 * x4
    h8 = (256.0 * x8 - 3584.0 * x6 + 13440.0 * x4
          - 13440.0 * x2 + 1680.0)
    h12 = (4096.0 * x12 - 212992.0 * x10 + 3317760.0 * x8
           - 25333440.0 * x6 + 100810080.0 * x4
           - 201620160.0 * x2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2),
    with a12 chosen so g(0)=0."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_exact(a, grid=None, refine=True):
    """Exact max of (1-g)/g over lambda>=0 on a dense grid, plus a local
    refinement near the argmax."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.0, 40.0, 200001),
            np.geomspace(1e-7, 80.0, 6000),
        ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    if refine and 0.0 < x0 < 80.0:
        local = np.linspace(max(0.0, x0 - 0.05), x0 + 0.05, 8001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-12, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        val = max(val, float(np.max(rl)))
    return val


def _surrogate(a, grid, temperature):
    """-softmin g  surrogate (log-sum-exp) with analytic gradient.

    softmin g = -t * log(mean(exp(-g/t)))
    so  F = -softmin g = t * log(mean(exp(-g/t)))
    dF/da = mean(exp(-g/t)) * (d(-g)/da) / mean(exp(-g/t))
          = sum_i w_i * (-dg_i/da),  w_i = exp(-g_i/t)/sum_j exp(-g_j/t)
    """
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(grid)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    expf = np.exp(-0.5 * grid * grid)
    g = poly * expf

    u = -g / temperature
    # shift for stability
    umax = float(np.max(u))
    e = np.exp(u - umax)
    s = float(np.sum(e))
    F = temperature * (math.log(s) - umax)

    w = e / s
    # dP/da_j (with a12 dependent on a):
    dh0 = h0 - (1.0 / 798336.0) * h12
    dh4 = h4 - (12.0 / 798336.0) * h12
    dh8 = h8 - (1680.0 / 798336.0) * h12
    # d(-g)/da_j = -expf * dP/da_j
    g0 = -float(np.sum(w * expf * dh0))
    g4 = -float(np.sum(w * expf * dh4))
    g8 = -float(np.sum(w * expf * dh8))
    return F, np.array([g0, g4, g8])


def construct():
    global _START
    if _START is None:
        _START = time.time()
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)

    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)

    from scipy.optimize import minimize, basin_hopping

    # Precompute grids for the surrogate (cheap, reused across calls).
    grid_fine = np.linspace(0.0, 40.0, 80001)
    grid_med = np.linspace(0.0, 30.0, 40001)

    def _improve(cand):
        nonlocal best_a, best_val
        cand = np.asarray(cand, dtype=float)
        if not np.all(np.isfinite(cand)):
            return
        v = _bound_exact(cand)
        if v < best_val:
            best_val = v
            best_a = cand.copy()

    # --- Phase 1: multi-start smooth surrogate (annealed temperature) ---
    rng = np.random.default_rng(42)
    starts = [_SEED.copy()]
    for scale in (0.05, 0.1, 0.2, 0.5):
        starts.append(_SEED + scale * 0.01 * rng.standard_normal(3))
    for _ in range(6):
        starts.append(rng.standard_normal(3) * 0.03)
    # hand-tuned neighbourhood of the seed
    for d0 in (-0.05, 0.05):
        for d4 in (-0.01, 0.01):
            for d8 in (-5e-5, 5e-5):
                starts.append(np.array([_SEED[0] + d0, _SEED[1] + d4,
                                        _SEED[2] + d8]))

    for temp in (0.05, 0.02, 0.005, 0.002):
        for s in starts:
            if remaining() < 3.0:
                break
            try:
                res = minimize(
                    lambda x, t=temp: _surrogate(x, grid_med, t)[0],
                    s,
                    jac=lambda x, t=temp: _surrogate(x, grid_med, t)[1],
                    method="L-BFGS-B",
                    options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-11},
                )
                _improve(res.x)
            except Exception:
                continue

    # --- Phase 2: basin-hopping global pass on the smooth surrogate ---
    if remaining() > 15.0:
        try:
            def bh_obj(x):
                return _surrogate(x, grid_med, 0.005)[0]
            state = basin_hopping(
                bh_obj,
                best_a,
                minimizer_kwargs={
                    "jac": lambda x: _surrogate(x, grid_med, 0.005)[1],
                    "method": "L-BFGS-B",
                    "options": {"maxiter": 200, "ftol": 1e-14, "gtol": 1e-11},
                },
                niter=40,
                stepsize=0.02,
                minimizer="L-BFGS-B",
                seed=1234,
            )
            _improve(state.x)
        except Exception:
            pass

    # --- Phase 3: high-precision polish on the exact (max) objective ---
    if remaining() > 5.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="L-BFGS-B",
                options={"maxiter": 1500, "ftol": 1e-15, "gtol": 1e-13},
            )
            _improve(res.x)
        except Exception:
            pass

    # --- Phase 4: adaptive local grid polish around the best point ---
    if remaining() > 2.0:
        for step in (0.003, 0.001, 0.0005, 0.0002):
            if remaining() < 1.0:
                break
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        if remaining() < 0.5:
                            break
                        cand = best_a + step * np.array([i, j, k], dtype=float)
                        _improve(cand)

    # --- Phase 5: final ultra-precise local solve (Nelder-Mead) ---
    if remaining() > 1.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-13, "fatol": 1e-15,
                         "adaptive": True},
            )
            _improve(res.x)
        except Exception:
            pass

    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
