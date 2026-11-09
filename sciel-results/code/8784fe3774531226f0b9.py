"""Optimized tuner for the Fourier-invariant Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

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
    x = np.linspace(-8.0, 8.0, 2001)
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
    """Create a closure for the objective function."""
    def objective(n):
        a0, a4, a8 = n[0], n[1], n[2]
        a12 = -(a0 + a4 / 6.0 + a8 / 108.0)
        
        poly = a0 * hs[0] + a4 * hs[1] + a8 * hs[2] + a12 * hs[3]
        gval = poly * g
        
        # L2 norm squared: integral |g|^2
        # L4 norm: (integral |g|^4)^{1/4}
        # L4 norm of f: (integral |f|^4)^{1/4}
        
        g2 = gval * gval
        g4 = g2 * g2
        
        # fval = sqrt(2*pi) * gval
        # f4 = (2*pi)^2 * g4
        # ||f||_4 = ( (2*pi)^2 * integral g4 )^{1/4} = sqrt(2*pi) * (integral g4)^{1/4} = sqrt(2*pi) * ||g||_4
        
        # C4 = ||g||_4 * ||f||_4 / ||g||_2
        #    = ||g||_4 * sqrt(2*pi) * ||g||_4 / ||g||_2
        #    = sqrt(2*pi) * ||g||_4^2 / ||g||_2
        
        # Let I2 = integral |g|^2, I4 = integral |g|^4
        # ||g||_2 = sqrt(I2)
        # ||g||_4 = I4^{1/4}
        # ||g||_4^2 = I4^{1/2} = sqrt(I4)
        
        # C4 = sqrt(2*pi) * sqrt(I4) / sqrt(I2) = sqrt(2*pi) * sqrt(I4 / I2)
        
        I2 = np.sum(g2) * dx
        I4 = np.sum(g4) * dx
        
        if I2 < 1e-12:
            return 1e10
            
        val = np.sqrt(2.0 * np.pi * I4 / I2)
        return val
    
    return objective


def construct():
    best_n = _SEED.copy()
    
    # Build grid
    x, hs, g, fhs = _build_grid()
    dx = x[1] - x[0]
    
    objective = _make_objective(hs, g, fhs, dx)
    
    best_val = objective(best_n)
    
    # Phase 1: Global Search with Differential Evolution
    # Bounds for a, b, c
    # a ~ -1.13, b ~ 0.04, c ~ 0.0003
    # Let's expand the search space significantly to ensure we don't miss better solutions.
    # But keep it reasonable to converge within time limits.
    bounds = [
        (-5.0, 5.0),
        (-1.0, 1.0),
        (-0.1, 0.1)
    ]
    
    if time.time() < _HARD_DEADLINE - 60.0:
        try:
            # Use differential evolution for global exploration
            # maxiter is number of generations
            # popsize is population size
            # tol is tolerance
            de_res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-6,
                seed=42,
                polish=False,  # We will do our own polish
                init='sobol'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_n = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local Refinement with Nelder-Mead
    # Start from the best point found by DE and the seed
    starts = [best_n.copy(), _SEED.copy()]
    
    # Add some random perturbations around the best point
    rng = np.random.default_rng(12345)
    for _ in range(10):
        pert = rng.normal(0, 0.1, 3)
        starts.append(best_n + pert)
    
    for s in starts:
        if time.time() > _HARD_DEADLINE - 10.0:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10,
                                    'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_n = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision Polish
    if time.time() < _HARD_DEADLINE - 5.0:
        try:
            res = minimize(objective, best_n, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12,
                                    'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_n = res.x.copy()
        except Exception:
            pass
    
    return best_n
# EVOLVE-BLOCK-END
