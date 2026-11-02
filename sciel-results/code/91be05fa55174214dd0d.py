"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the classical upper bound on A(f)*A(Fourier(f)) over the three free
Hermite coefficients (degrees 0, 4, 8).  The degree-12 weight is fixed so that
g(0) = 0.  We minimize the bound
    B(a) = max_{lambda>=0} (1 - g(lambda)) / g(lambda)
using a smooth log-sum-exp surrogate for the outer max during optimization,
then a high-precision polish on the exact (max) objective.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 25.0          # safety margin (seconds) before the hard kill
_START = None

# Precompute grids and Hermite basis for speed
def _init_grids():
    """Precompute fixed grids and their Hermite basis values."""
    # Main evaluation grid: dense near origin where action happens
    grid_main = np.concatenate([
        np.linspace(0.0, 8.0, 50001),       # dense near origin
        np.linspace(8.0, 20.0, 20001),      # medium density
        np.linspace(20.0, 40.0, 5001),      # sparse tail
    ])
    # Smooth surrogate grid (slightly coarser for speed)
    grid_smooth = np.linspace(0.0, 25.0, 30001)
    
    # Precompute Hermite polynomials on these grids
    h0_main, h4_main, h8_main, h12_main = _hermite_values(grid_main)
    h0_smooth, h4_smooth, h8_smooth, h12_smooth = _hermite_values(grid_smooth)
    
    # Precompute exp(-lambda^2/2)
    exp_main = np.exp(-0.5 * grid_main * grid_main)
    exp_smooth = np.exp(-0.5 * grid_smooth * grid_smooth)
    
    return {
        'main': (grid_main, h0_main, h4_main, h8_main, h12_main, exp_main),
        'smooth': (grid_smooth, h0_smooth, h4_smooth, h8_smooth, h12_smooth, exp_smooth)
    }

def _h2n(n):
    """H_0(0), H_4(0), H_8(0), H_12(0) for the physicists' Hermite polynomials."""
    return 1.0, 12.0, 1680.0, 798336.0


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    x2 = x * x
    h4 = 16.0 * x2 * x2 - 48.0 * x2 + 12.0
    h8 = 256.0 * x2**4 - 3584.0 * x2**3 + 13440.0 * x2**2 - 13440.0 * x2 + 1680.0
    h12 = (4096.0 * x2**6 - 212992.0 * x2**5 + 3317760.0 * x2**4
           - 25333440.0 * x2**3 + 100810080.0 * x2**2 - 201620160.0 * x2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2),
    with a12 chosen so g(0)=0."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _g_fast(a, grid_data):
    """Fast evaluation of g using precomputed Hermite basis."""
    grid, h0, h4, h8, h12, expf = grid_data
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * expf


def _bound_exact_fast(a, grid_data):
    """Exact max of (1-g)/g using precomputed grid."""
    g = _g_fast(a, grid_data)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    # Local refinement
    grid = grid_data[0]
    x0 = float(grid[idx])
    if 0.0 < x0 < 40.0:
        local = np.linspace(max(0.0, x0 - 0.02), x0 + 0.02, 2001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-12, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        val = max(val, float(np.max(rl)))
    return val


def _bound_smooth_fast(a, grid_data, temperature=0.015):
    """Smooth log-sum-exp surrogate using precomputed grid."""
    g = _g_fast(a, grid_data)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    softmax = m + temperature * math.log(float(np.mean(s)))
    return softmax


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Initialize precomputed grids
    grids = _init_grids()
    main_grid = grids['main']
    smooth_grid = grids['smooth']
    
    best_a = _SEED.copy()
    best_val = _bound_exact_fast(_SEED, main_grid)

    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)

    from scipy.optimize import minimize
    rng = np.random.default_rng(42)
    
    # --- Phase 1: Multi-start global optimization on smooth surrogate ---
    # Generate diverse starting points
    starts = [_SEED.copy()]
    
    # Structured perturbations around seed
    for scale in (0.01, 0.02, 0.05, 0.1):
        for _ in range(3):
            starts.append(_SEED + scale * rng.standard_normal(3))
    
    # Random starts in reasonable range
    for _ in range(10):
        a0 = rng.uniform(-2.0, 0.5)
        a4 = rng.uniform(-0.1, 0.2)
        a8 = rng.uniform(-0.005, 0.01)
        starts.append(np.array([a0, a4, a8]))
    
    # Physics-inspired starts: try to shape g to minimize peak of (1-g)/g
    # We want g(lambda) to be close to 1 where it matters, but g(0)=0
    # Try combinations that might flatten the ratio
    starts.append(np.array([-1.0, 0.04, 0.0008]))
    starts.append(np.array([-1.2, 0.05, 0.0012]))
    starts.append(np.array([-0.8, 0.03, 0.0006]))
    
    # Optimize each start on smooth surrogate
    for s in starts:
        if remaining() < 5.0:
            break
        try:
            res = minimize(
                lambda x: _bound_smooth_fast(x, smooth_grid),
                s, method="L-BFGS-B",
                options={"maxiter": 150, "ftol": 1e-10, "gtol": 1e-6},
            )
            cand = np.asarray(res.x, dtype=float)
            if not np.all(np.isfinite(cand)):
                continue
            v = _bound_exact_fast(cand, main_grid)
            if v < best_val:
                best_val = v
                best_a = cand.copy()
        except Exception:
            continue
    
    # --- Phase 2: High-precision polish on exact objective ---
    if remaining() > 4.0:
        try:
            res = minimize(
                lambda x: _bound_exact_fast(x, main_grid),
                best_a, method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact_fast(cand, main_grid)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    # --- Phase 3: Systematic local polish with adaptive step ---
    if remaining() > 3.0:
        # Start with larger steps, then refine
        for step_scale in (0.01, 0.005, 0.002, 0.001):
            if remaining() < 2.0:
                break
            step = step_scale
            # Try perturbations in each coordinate and combinations
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        if remaining() < 1.5:
                            break
                        if i == 0 and j == 0 and k == 0:
                            continue
                        cand = best_a + step * np.array([i, j, k], dtype=float)
                        v = _bound_exact_fast(cand, main_grid)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()
    
    # --- Phase 4: Final Nelder-Mead polish for robustness ---
    if remaining() > 2.0:
        try:
            res = minimize(
                lambda x: _bound_exact_fast(x, main_grid),
                best_a, method="Nelder-Mead",
                options={"maxiter": 200, "xatol": 1e-10, "fatol": 1e-10},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact_fast(cand, main_grid)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass
    
    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
