"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound C4 = max_{lambda>=0} (1-g(lambda))/g(lambda)
where g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2)
and a12 is fixed so g(0) = 0.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 20.0
_START = None


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    h4 = 16.0 * x4 - 48.0 * x2 + 12.0
    h8 = 256.0 * x8 - 3584.0 * x6 + 13440.0 * x4 - 13440.0 * x2 + 1680.0
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


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 with adaptive refinement."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.0, 5.0, 50001),
            np.linspace(5.0, 15.0, 30000),
            np.linspace(15.0, 40.0, 10000),
            np.geomspace(1e-8, 0.01, 1000),
        ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    for level in range(3):
        half = 0.02 / (level + 1)
        lo = max(0.0, x0 - half)
        hi = x0 + half
        local = np.linspace(lo, hi, 2001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-14, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        idx2 = int(np.argmax(rl))
        val = max(val, float(rl[idx2]))
        x0 = float(local[idx2])
    return val


def _objective_ratio(a, grid, beta=30.0):
    """Smooth surrogate for max(1-g)/g using soft-max via log-sum-exp."""
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-10, ratio, 1e15)
    ratio = np.where(np.isfinite(ratio), ratio, 1e15)
    ratio = np.clip(ratio, -20.0, 1e12)
    r_over_beta = ratio / beta
    m = np.max(r_over_beta)
    s = np.exp(r_over_beta - m)
    soft_max_val = beta * (m + math.log(float(np.mean(s))))
    return soft_max_val


def _objective_ratio_grad(a, grid, beta=30.0):
    """Gradient of _objective_ratio w.r.t. a = [a0, a4, a8]."""
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-10, ratio, 1e15)
    ratio = np.where(np.isfinite(ratio), ratio, 1e15)
    ratio = np.clip(ratio, -20.0, 1e12)
    
    r_over_beta = ratio / beta
    m = np.max(r_over_beta)
    exp_vals = np.exp(r_over_beta - m)
    mean_exp = float(np.mean(exp_vals))
    weights = exp_vals / mean_exp
    
    h0, h4, h8, h12 = _hermite_values(grid)
    exp_factor = np.exp(-0.5 * grid * grid)
    
    dg_da0 = (h0 - h12 / 798336.0) * exp_factor
    dg_da4 = (h4 - 12.0 * h12 / 798336.0) * exp_factor
    dg_da8 = (h8 - 1680.0 * h12 / 798336.0) * exp_factor
    
    with np.errstate(divide="ignore", invalid="ignore"):
        dratio_dg = np.where(g > 1e-10, -1.0 / (g * g), 0.0)
    
    grad = np.zeros(3)
    grad[0] = float(np.dot(weights * dratio_dg / beta, dg_da0))
    grad[1] = float(np.dot(weights * dratio_dg / beta, dg_da4))
    grad[2] = float(np.dot(weights * dratio_dg / beta, dg_da8))
    
    soft_max_val = beta * (m + math.log(mean_exp))
    return soft_max_val, grad


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)
    
    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)
    
    from scipy.optimize import minimize, differential_evolution
    
    # Grid for optimization
    grid_opt = np.concatenate([
        np.linspace(0.001, 5.0, 20000),
        np.linspace(5.0, 12.0, 10000),
    ])
    
    # --- Phase 1: Global search with differential evolution ---
    if remaining() > 50.0:
        try:
            bounds = [(-3.0, 0.5), (-0.5, 0.5), (-0.01, 0.01)]
            result_de = differential_evolution(
                lambda x: _objective_ratio(x, grid_opt, beta=30.0),
                bounds=bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=25,
                polish=True,
                init="sobol",
            )
            cand = np.asarray(result_de.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    if remaining() > 40.0:
        try:
            bounds = [(-3.0, 0.5), (-0.5, 0.5), (-0.01, 0.01)]
            result_de2 = differential_evolution(
                lambda x: _objective_ratio(x, grid_opt, beta=20.0),
                bounds=bounds,
                seed=123,
                maxiter=150,
                tol=1e-8,
                popsize=20,
                polish=True,
            )
            cand = np.asarray(result_de2.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    if remaining() > 35.0:
        try:
            bounds = [(-1.5, -0.8), (-0.2, 0.2), (-0.005, 0.005)]
            result_de3 = differential_evolution(
                lambda x: _objective_ratio(x, grid_opt, beta=50.0),
                bounds=bounds,
                seed=777,
                maxiter=200,
                tol=1e-9,
                popsize=30,
                polish=True,
            )
            cand = np.asarray(result_de3.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # --- Phase 2: Multi-start local optimization with gradients ---
    rng = np.random.default_rng(0)
    starts = [best_a.copy(), _SEED.copy()]
    for scale in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        starts.append(best_a + scale * rng.standard_normal(3))
        starts.append(_SEED + scale * rng.standard_normal(3))
    for i in range(3):
        for sign in (-1, 1):
            for scale in (0.01, 0.03, 0.1, 0.3):
                p = best_a.copy()
                p[i] += sign * scale
                starts.append(p)
    
    hand_picks = [
        [-1.13, 0.04, 0.0003],
        [-1.10, 0.035, 0.0004],
        [-1.15, 0.045, 0.0002],
        [-1.05, 0.03, 0.0005],
        [-1.20, 0.05, 0.0001],
        [-1.00, 0.025, 0.0006],
        [-1.08, 0.038, 0.00035],
        [-1.12, 0.042, 0.00025],
        [-1.18, 0.048, 0.00015],
        [-1.25, 0.055, 0.00005],
    ]
    for hp in hand_picks:
        starts.append(np.array(hp, dtype=float))
    
    for temp_beta in (50.0, 30.0, 20.0, 10.0):
        for s in starts:
            if remaining() < 5.0:
                break
            try:
                res = minimize(
                    lambda x: _objective_ratio(x, grid_opt, beta=temp_beta),
                    s,
                    jac=lambda x: _objective_ratio_grad(x, grid_opt, beta=temp_beta)[1],
                    method="L-BFGS-B",
                    options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12},
                )
            except Exception:
                continue
            cand = np.asarray(res.x, dtype=float)
            if not np.all(np.isfinite(cand)):
                continue
            v = _bound_exact(cand)
            if v < best_val:
                best_val = v
                best_a = cand.copy()
    
    # --- Phase 3: High-precision polish on exact objective ---
    if remaining() > 20.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-14},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    if remaining() > 15.0:
        pert_dirs = [
            np.array([1.0, -1.0, 0.5]),
            np.array([-0.5, 1.0, 1.0]),
            np.array([0.5, 0.5, -1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        for pert_dir in pert_dirs:
            if remaining() < 5.0:
                break
            try:
                pert = best_a + 0.002 * pert_dir
                res2 = minimize(
                    _bound_exact, pert, method="L-BFGS-B",
                    options={"maxiter": 1500, "ftol": 1e-16, "gtol": 1e-14},
                )
                cand = np.asarray(res2.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
            except Exception:
                pass
    
    # --- Phase 4: Fine local search with coordinate descent ---
    if remaining() > 10.0:
        for step in (0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005):
            if remaining() < 3.0:
                break
            improved = True
            while improved and remaining() > 3.0:
                improved = False
                for dim in range(3):
                    for sign in (-1, 1):
                        if remaining() < 1.0:
                            break
                        cand = best_a.copy()
                        cand[dim] += sign * step
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()
                            improved = True
    
    # --- Phase 5: Nelder-Mead polish ---
    if remaining() > 5.0:
        try:
            res_nm = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-14, "fatol": 1e-16, "adaptive": True},
            )
            cand = np.asarray(res_nm.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # --- Phase 6: Final coordinate descent ---
    if remaining() > 3.0:
        for _ in range(100):
            if remaining() < 1.0:
                break
            improved = False
            for dim in range(3):
                for delta in (0.0005, -0.0005, 0.0002, -0.0002, 0.0001, -0.0001, 0.00005, -0.00005):
                    if remaining() < 0.5:
                        break
                    cand = best_a.copy()
                    cand[dim] += delta
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
                        improved = True
            if not improved:
                break
    
    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
