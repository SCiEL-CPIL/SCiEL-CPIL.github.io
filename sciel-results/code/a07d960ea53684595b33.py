"""Optimized tuner for the Fourier-invariant Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

_T0 = time.time()
_HARD_DEADLINE = 190.0  # safety margin before the 210 s kill
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def _h(x, n):
    """Physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x, dtype=float)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _coeffs(a, b, c):
    """Full coefficient vector [H0, H4, H8, H12] with g(0)=0 enforced."""
    d = -(a + b / 6.0 + c / 108.0)
    return np.array([a, b, c, d])


def _build_grid():
    """Build a fine frequency grid for the upper bound."""
    x = np.linspace(-10.0, 10.0, 4001)
    hs = [_h(x, k) for k in (0, 4, 8, 12)]
    g = np.exp(-0.5 * x * x)
    fhs = hs  # same basis for the Fourier side (Fourier-invariant structure)
    return x, hs, g, fhs


def _compute_bound_vec(n, hs, g, fhs):
    """
    Compute the uncertainty product upper bound for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by the constraint g(0) = 0.
    
    The quantity we want is C4 = ||g||_4 * ||\hat{g}||_4 / ||g||_2.
    Due to Fourier invariance of the Hermite-Gaussian basis and the specific structure,
    we compute this via numerical integration.
    """
    a0, a4, a8 = n[0], n[1], n[2]
    a12 = -(a0 + a4 / 6.0 + a8 / 108.0)
    
    # Construct g(x)
    poly = a0 * hs[0] + a4 * hs[1] + a8 * hs[2] + a12 * hs[3]
    gval = poly * g
    
    # Construct \hat{g}(x)
    # Fourier transform of H_n(x) exp(-x^2/2) is sqrt(2*pi) * (-i)^n H_n(x) exp(-x^2/2)
    # Wait, the standard unitary Fourier transform is usually defined with 1/sqrt(2*pi).
    # The problem statement says "Fourier-invariant".
    # Let's assume the standard math convention where F[f](x) = (1/sqrt(2*pi)) int f(t) e^{-ixt} dt.
    # For Hermite functions psi_n(x) = H_n(x) exp(-x^2/2), F[psi_n] = (-i)^n psi_n.
    # So F[g] = sum c_n (-i)^n psi_n.
    # However, the seed program used `scale = np.sqrt(2.0 * np.pi)` and no phase.
    # Let's look at the seed program's `_value` or `_objective`.
    # Seed program:
    # fval = g * sum(fcoeffs[i] * fhs[i] ...)
    # fcoeffs = [scale * c[i] ...]
    # This implies F[H_n exp(-x^2/2)] = sqrt(2*pi) * H_n exp(-x^2/2) ??
    # This corresponds to the non-unitary Fourier transform F[f](x) = int f(t) e^{-ixt} dt.
    # In that case, F[exp(-x^2/2)] = sqrt(2*pi) exp(-x^2/2).
    # F[H_n(x) exp(-x^2/2)] = (-i)^n sqrt(2*pi) H_n(x) exp(-x^2/2).
    # The phases (-i)^n for n=0,4,8,12 are all 1. So the sign is positive.
    # So fval = sqrt(2*pi) * (a0*H0 + a4*H4 + a8*H8 + a12*H12) * g.
    # This is exactly what the seed program did.
    
    scale = np.sqrt(2.0 * np.pi)
    fval = scale * poly * g
    
    # Compute norms
    # We need ||g||_2, ||g||_4, ||f||_4.
    # Note: dx is needed for integration.
    # The grid is uniform.
    # The caller should pass dx or we assume it's handled.
    # To make this function self-contained for optimization, we should pass dx.
    # But for speed in the inner loop, it's better to precompute weights if possible.
    # However, the grid is fixed.
    
    # Let's assume the caller provides dx.
    # Actually, let's modify the signature to include dx.
    pass


def _make_objective(hs, g, fhs, dx):
    """Create a closure for the objective function with numerical safeguards."""
    def objective(n):
        a0, a4, a8 = n[0], n[1], n[2]
        a12 = -(a0 + a4 / 6.0 + a8 / 108.0)
        
        poly = a0 * hs[0] + a4 * hs[1] + a8 * hs[2] + a12 * hs[3]
        gval = poly * g
        
        g2 = gval * gval
        g4 = g2 * g2
        
        I2 = np.sum(g2) * dx
        I4 = np.sum(g4) * dx
        
        if I2 < 1e-15:
            return 1e10
        
        # Use log-space for numerical stability when I4/I2 is very small or large
        log_ratio = np.log(I4) - np.log(I2)
        if log_ratio > 500:  # overflow guard
            return 1e10
        if log_ratio < -500:  # underflow guard
            return 0.0
            
        val = np.sqrt(2.0 * np.pi) * np.exp(0.5 * log_ratio)
        return val
    
    return objective


def construct():
    best_n = _SEED.copy()
    
    # Build grid
    x, hs, g, fhs = _build_grid()
    dx = x[1] - x[0]
    
    objective = _make_objective(hs, g, fhs, dx)
    
    best_val = objective(best_n)
    
    # Phase 1: Focused global search around seed with tighter bounds
    # The seed is likely near optimal; use moderate bounds to avoid wasted computation
    bounds = [
        (-3.0, 0.0),   # a0: negative, moderate range
        (-0.5, 0.5),   # a4: small
        (-0.05, 0.05)  # a8: very small
    ]
    
    if time.time() < _HARD_DEADLINE - 45.0:
        try:
            # Use differential evolution for global exploration with tighter bounds
            de_res = differential_evolution(
                objective, 
                bounds, 
                maxiter=30, 
                popsize=20, 
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_n = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start local refinement
    starts = [best_n.copy(), _SEED.copy()]
    
    # Add perturbations around both best point and seed
    rng = np.random.default_rng(12345)
    for _ in range(15):
        pert = rng.normal(0, 0.05, 3)
        starts.append(best_n + pert)
    for _ in range(5):
        pert = rng.normal(0, 0.1, 3)
        starts.append(_SEED + pert)
    
    for s in starts:
        if time.time() > _HARD_DEADLINE - 15.0:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12,
                                    'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_n = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision final polish with tighter tolerances
    if time.time() < _HARD_DEADLINE - 5.0:
        try:
            res = minimize(objective, best_n, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14,
                                    'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_n = res.x.copy()
        except Exception:
            pass
    
    return best_n
# EVOLVE-BLOCK-END
