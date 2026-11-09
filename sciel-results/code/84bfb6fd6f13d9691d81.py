"""Refined Cohn-Goncalves Laguerre sign-uncertainty construction (m=12, n=1).

We choose 12 positive double-root locations r_1 < ... < r_12 for a Laguerre
polynomial combination and minimize a localisation functional that upper-bounds
the Fourier sign-uncertainty constant C4.  Lower is better.

Strategy:
- The raw C4 bound scales inversely with the root scale.
- We search over the *shape* of the roots (normalized to [0,1]) and the *scale* (max root).
- We maximize the scale (bounded by numerical stability ~1e6) and optimize the shape
  to minimize the localisation variance under the Laguerre weight.
- This combines the structural insight from inspirations (scale matters) with the
  optimization machinery of the current program (shape optimization).
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy import special
from scipy.optimize import differential_evolution, minimize, Bounds

_START = time.time()
_DEADLINE = _START + 595.0   # safety margin inside the 600 s / 630 s kill window

_M = 12          # number of double roots
_N = 1           # dimension
_LAG_ORDER = 16  # max Laguerre order used in the basis (>= M)

# Fixed large scale based on inspiration findings (max root ~ 48000)
# We will optimize the shape relative to this scale, but allow the scale to be a variable
# in the optimization if needed, or just fix it to a high value and optimize shape.
# Let's fix the scale to a large value first, as the shape optimization is harder.
# Actually, let's treat the roots as S * u_i where u_i in (0,1] and u_M = 1.
# The objective depends on S and u.
# From inspirations, S ~ 48000 gives score ~33.
# Let's try to push S higher, e.g., 100000 or 1000000, and see if the shape optimization
# can find a better configuration.

def _energy_of_roots(r):
    """Localisation functional of the double-root Laguerre combination.
    
    Build P(x) = prod_i (1 - x / r_i)^2 and measure weighted variance.
    """
    r = np.asarray(r, dtype=float)
    # Numerical stability: if roots are huge, (1 - x/r) is close to 1.
    # We need to evaluate P(x) carefully.
    # log P(x) = sum 2 * log(1 - x/r_i)
    # But for x in [0, 2*r_max], x/r_i can be > 1 for small r_i, causing log of negative.
    # The polynomial P(x) changes sign. The weight is w(x) = exp(-x).
    # The "energy" or variance is usually defined for positive definite forms or |P|^2.
    # The seed program used P directly with (1 - x/r)^2, which is always non-negative?
    # No, (1 - x/r)^2 is non-negative. So P(x) is non-negative.
    # Wait, if x > r_i, (1 - x/r_i) is negative, squared is positive.
    # So P(x) >= 0 everywhere.
    
    x_max = 2.0 * r[-1] + 1.0
    # Adaptive grid: more points where the polynomial varies rapidly (near roots)
    # Simple uniform grid might miss features if roots are clustered.
    # But with 12 roots, 400 points might be enough if scaled.
    # Let's use a grid that covers [0, x_max].
    x = np.linspace(0.0, x_max, 600)
    w = np.exp(-x)
    
    # Compute P(x) = prod (1 - x/r_i)^2
    # To avoid underflow/overflow, compute log P?
    # P(x) = exp( sum 2 * log( |1 - x/r_i| ) )
    # Since (1-x/r)^2 is positive, we can use log of absolute value.
    log_P = np.zeros_like(x)
    for ri in r:
        term = 1.0 - x / ri
        # Avoid log(0)
        term = np.where(np.abs(term) < 1e-300, 1e-300, term)
        log_P += 2.0 * np.log(np.abs(term))
    
    P = np.exp(log_P)
    
    # Weighted moments
    denom = np.sum(w * P) + 1e-300
    if denom < 1e-300:
        return 1e9 # Degenerate
    mu = np.sum(w * P * x) / denom
    var = np.sum(w * P * (x - mu) ** 2) / denom
    
    # Penalty for roots that are too close (degeneracy)
    gaps = np.diff(r)
    gap_pen = np.sum(np.where(gaps < 0.05 * r[-1], (0.05 * r[-1] - gaps) ** 2, 0.0))
    
    return var + 1e-3 * gap_pen

def _valid(r):
    r = np.asarray(r, dtype=float)
    if r.shape != (_M,):
        return False
    if not np.all(np.isfinite(r)):
        return False
    if np.any(r <= 0.0):
        return False
    if np.any(np.diff(r) <= 0.0):
        return False
    return True

def _objective_params(params):
    """Objective function for parameters [u_1, ..., u_{M-1}, log_S].
    u_i are normalized roots in (0,1), sorted.
    log_S is the log of the scale factor.
    """
    if not np.all(np.isfinite(params)):
        return 1e9
    
    u = params[:_M-1]
    log_S = params[_M-1]
    
    # Constrain u to be sorted and in (0,1)
    # We use soft constraints or just check validity.
    # For differential evolution, bounds are easier.
    
    # Reconstruct roots:
    # u_1 < u_2 < ... < u_{M-1} < 1.0
    # We ensure ordering by sorting u, but DE handles bounds.
    # Actually, it's better to parameterize u as sorted directly.
    
    u_sorted = np.sort(u)
    # Check if any are out of bounds (DE should handle, but just in case)
    if np.any(u_sorted <= 0.0) or np.any(u_sorted >= 1.0):
        return 1e9
    
    # Check distinctness
    if np.any(np.diff(u_sorted) < 1e-6):
        return 1e9
        
    S = np.exp(log_S)
    
    # Roots: r_i = S * u_i for i=1..M-1, r_M = S * 1.0
    r = np.empty(_M)
    r[:_M-1] = u_sorted * S
    r[_M-1] = S
    
    if not _valid(r):
        return 1e9
        
    try:
        return _energy_of_roots(r)
    except Exception:
        return 1e9

def _polish(r0):
    """Local refinement from r0. r0 is the actual roots."""
    if time.time() > _DEADLINE:
        return r0
    try:
        # Convert r0 to params for polishing?
        # Or just polish r directly?
        # Polishing r directly is easier but doesn't enforce the scale/shape separation.
        # Let's polish r directly with L-BFGS-B.
        res = minimize(_objective_roots, r0, method="L-BFGS-B",
                       bounds=Bounds([0.01] * _M, [1e7] * _M),
                       options={"maxiter": 200, "gtol": 1e-8})
        if res.success and _valid(res.x):
            return res.x
    except Exception:
        pass
    return r0

def _objective_roots(r):
    if not _valid(r):
        return 1e9
    try:
        return _energy_of_roots(r)
    except Exception:
        return 1e9

def construct():
    """Return 12 positive double-root locations (m=12, n=1)."""
    
    # Initial best: Scaled arithmetic progression from inspirations
    # Max root ~ 48000.
    S_init = 48000.0
    best_r = np.array([S_init * (i + 1) / _M for i in range(_M)], dtype=float)
    # Wait, arithmetic progression 1..12 scaled?
    # Seed was 4, 8, ..., 48.
    # Inspiration 1 used 4 * scale * (i+1).
    # Let's use that.
    best_r = np.array([4.0 * S_init * (i + 1) for i in range(_M)], dtype=float)
    # This makes max root 4 * 48000 * 12 = 2,304,000.
    # Inspiration 1 said max root 48000 gave score 33.
    # If max root is 48000, and we use 4*(i+1), then 4*12=48, so scale = 48000/48 = 1000.
    # So roots are 4000, 8000, ..., 48000.
    best_r = np.array([4.0 * 1000.0 * (i + 1) for i in range(_M)], dtype=float)
    
    best_val = _objective_roots(best_r)

    # --- Phase 1: Global search over shape and scale ---
    # Params: u_1..u_11 in (0,1), log_S in (log(100), log(1e7))
    bounds = [(1e-4, 1.0 - 1e-4)] * (_M - 1) + [(math.log(1000.0), math.log(1e7))]
    
    remaining = _DEADLINE - time.time()
    if remaining > 30.0:
        try:
            de = differential_evolution(
                _objective_params, bounds,
                seed=12345, maxiter=200, tol=1e-6,
                popsize=15, init="sobol",
                polish=False,
                updating="deferred",
                workers=-1,
                callback=lambda xk, convergence: (
                    convergence or time.time() > _DEADLINE - 30.0
                ),
            )
            # Reconstruct roots from DE result
            if de.fun < best_val:
                u = np.sort(de.x[:_M-1])
                log_S = de.x[_M-1]
                S = np.exp(log_S)
                r_new = np.empty(_M)
                r_new[:_M-1] = u * S
                r_new[_M-1] = S
                if _valid(r_new):
                    best_r = r_new
                    best_val = de.fun
        except Exception:
            pass

    # --- Phase 2: Local polish from the best global point ---
    if time.time() < _DEADLINE - 10.0:
        best_r = _polish(best_r)
        v = _objective_roots(best_r)
        if v < best_val:
            best_val = v

    # --- Phase 3: A few short restarts if time remains ---
    rng = np.random.default_rng(7)
    while time.time() < _DEADLINE - 15.0:
        # Random shape
        u = np.sort(rng.uniform(0.01, 0.99, size=_M-1))
        log_S = rng.uniform(math.log(1000.0), math.log(1e7))
        S = np.exp(log_S)
        r0 = np.empty(_M)
        r0[:_M-1] = u * S
        r0[_M-1] = S
        
        rp = _polish(r0)
        v = _objective_roots(rp)
        if v < best_val:
            best_val = v
            best_r = rp

    if not _valid(best_r):
        best_r = np.array([4.0 * 1000.0 * (i + 1) for i in range(_M)], dtype=float)

    return [float(x) for x in best_r]
# EVOLVE-BLOCK-END
