"""Uncertainty inequality optimization exploiting homogeneity to reduce to 1D search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, brent

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..24."""
    I = np.zeros(25)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 25):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _coeffs_from_ratios(r4, r8):
    """
    Compute polynomial coefficients p_k of P(x) = sum p_k x^(2k)
    given ratios r4 = a4/a0, r8 = a8/a0, with a0 = 1 (homogeneity).
    a12 = -(1 + 3*r4 + 105*r8)/945
    """
    a12 = -(1.0 + 3.0 * r4 + 105.0 * r8) / 945.0
    p = np.zeros(7)
    p[0] = 0.0  # constraint satisfied by construction
    p[1] = -6.0 * r4 - 420.0 * r8 - 7560.0 * a12
    p[2] = r4 + 210.0 * r8 + 13860.0 * a12
    p[3] = -28.0 * r8 - 6930.0 * a12
    p[4] = r8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


def _compute_bound_from_ratios(r4, r8):
    """
    Compute the uncertainty product sqrt(vf * vF) given ratios r4, r8.
    Due to homogeneity, the scale factor cancels, so we can set a0=1.
    """
    p = _coeffs_from_ratios(r4, r8)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    m = np.zeros(3)
    n = np.zeros(3)
    
    for i in range(7):
        for j in range(7):
            pij = p[i] * p[j]
            qij = q[i] * q[j]
            if pij == 0.0 and qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 25:
                    m[k] += pij * _I[idx]
                    n[k] += qij * _I[idx]
    n *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 <= 0 or n2 <= 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18
    
    return bound


def _objective_2d(params):
    """Objective for 2D optimization over (r4, r8)."""
    return _compute_bound_from_ratios(params[0], params[1])


def construct():
    best_val = np.inf
    best_ratios = np.array([0.04 / -1.13, 0.000308642 / -1.13])  # from seed: a4/a0, a8/a0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: High-resolution grid search over (r4, r8)
    # r4 = a4/a0, r8 = a8/a0
    # From seed: a0 ≈ -1.13, a4 ≈ 0.04, a8 ≈ 0.000308642
    # So r4 ≈ -0.035, r8 ≈ -0.000273
    
    r4_min, r4_max = -0.5, 0.5
    r8_min, r8_max = -0.02, 0.02
    
    # Coarse grid first
    n_coarse = 50
    r4_vals = np.linspace(r4_min, r4_max, n_coarse)
    r8_vals = np.linspace(r8_min, r8_max, n_coarse)
    
    best_grid_val = np.inf
    best_grid_r4 = r4_vals[0]
    best_grid_r8 = r8_vals[0]
    
    for i, r4 in enumerate(r4_vals):
        if time.time() > _DEADLINE - 120:
            break
        for r8 in r8_vals:
            val = _compute_bound_from_ratios(r4, r8)
            if val < best_grid_val:
                best_grid_val = val
                best_grid_r4 = r4
                best_grid_r8 = r8
    
    if best_grid_val < best_val:
        best_val = best_grid_val
        best_ratios = np.array([best_grid_r4, best_grid_r8])
    
    # Phase 2: Fine grid around the best coarse point
    if time.time() < _DEADLINE - 100:
        r4_fine_min = best_grid_r4 - 0.02
        r4_fine_max = best_grid_r4 + 0.02
        r8_fine_min = best_grid_r8 - 0.005
        r8_fine_max = best_grid_r8 + 0.005
        
        n_fine = 100
        r4_fine_vals = np.linspace(r4_fine_min, r4_fine_max, n_fine)
        r8_fine_vals = np.linspace(r8_fine_min, r8_fine_max, n_fine)
        
        for r4 in r4_fine_vals:
            if time.time() > _DEADLINE - 80:
                break
            for r8 in r8_fine_vals:
                val = _compute_bound_from_ratios(r4, r8)
                if val < best_val:
                    best_val = val
                    best_ratios = np.array([r4, r8])
    
    # Phase 3: Brent's method on 1D slices for local refinement
    if time.time() < _DEADLINE - 60:
        # For several r4 values near the best, optimize over r8 using Brent
        r4_candidates = np.linspace(best_ratios[0] - 0.01, best_ratios[0] + 0.01, 20)
        
        for r4 in r4_candidates:
            if time.time() > _DEADLINE - 50:
                break
            try:
                def obj_r8(r8):
                    return _compute_bound_from_ratios(r4, r8)
                
                res = brent(obj_r8, brack=(r8_min, best_ratios[1], r8_max),
                           tol=1e-15, xtol=1e-15)
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_ratios = np.array([r4, res.x])
            except Exception:
                pass
        
        # For several r8 values near the best, optimize over r4 using Brent
        r8_candidates = np.linspace(best_ratios[1] - 0.003, best_ratios[1] + 0.003, 20)
        
        for r8 in r8_candidates:
            if time.time() > _DEADLINE - 40:
                break
            try:
                def obj_r4(r4):
                    return _compute_bound_from_ratios(r4, r8)
                
                res = brent(obj_r4, brack=(r4_min, best_ratios[0], r4_max),
                           tol=1e-15, xtol=1e-15)
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_ratios = np.array([res.x, r8])
            except Exception:
                pass
    
    # Phase 4: 2D L-BFGS-B polish from best point
    if time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective_2d, best_ratios, method='L-BFGS-B',
                          options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_ratios = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision Brent refinement
    if time.time() < _DEADLINE - 5:
        try:
            # Final 1D optimization over r4 with r8 fixed
            def obj_r4_final(r4):
                return _compute_bound_from_ratios(r4, best_ratios[1])
            
            res = brent(obj_r4_final, brack=(best_ratios[0] - 0.005, best_ratios[0], best_ratios[0] + 0.005),
                       tol=1e-16, xtol=1e-16)
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_ratios = np.array([res.x, best_ratios[1]])
            
            # Final 1D optimization over r8 with r4 fixed
            def obj_r8_final(r8):
                return _compute_bound_from_ratios(best_ratios[0], r8)
            
            res = brent(obj_r8_final, brack=(best_ratios[1] - 0.002, best_ratios[1], best_ratios[1] + 0.002),
                       tol=1e-16, xtol=1e-16)
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_ratios = np.array([best_ratios[0], res.x])
        except Exception:
            pass
    
    # Convert back to (a0, a4, a8) with a0 = -1.13 (scale doesn't matter due to homogeneity)
    a0 = -1.13
    a4 = a0 * best_ratios[0]
    a8 = a0 * best_ratios[1]
    
    return np.array([a0, a4, a8])
# EVOLVE-BLOCK-END
