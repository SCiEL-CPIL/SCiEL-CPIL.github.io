"""Combined search for the Uncertainty Inequality task."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Deadline: evaluator kills at 210s. Use 195s budget to be safe.
_BUDGET_SECONDS = 195.0

# Pre-computed Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = 16*x^4 - 48*x^2 + 12
# H8(x) = 256*x^8 - 4480*x^6 + 24192*x^4 - 40320*x^2 + 1680
# H12(x) = 4096*x^12 - 147456*x^10 + 1684608*x^8 - 9067200*x^6 + 20901120*x^4 - 20321280*x^2 + 665280

# Values at x=0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

def _compute_value(c0, c1, c2):
    """
    Compute the value = r^2 / (2*pi) for the given coefficients.
    Returns a positive float (lower is better), or inf if invalid.
    """
    try:
        # c3 is fixed so that g(0) = 0
        c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
        
        # Coefficients for g(x)
        a2 = c1*(-48) + c2*(-40320) + c3*(-20321280)
        a4 = c1*16 + c2*24192 + c3*20901120
        a6 = c2*(-4480) + c3*(-9067200)
        a8 = c2*256 + c3*1684608
        a10 = c3*(-147456)
        a12 = c3*4096
        
        # If c3 < 0, flip all coefficients (g -> -g)
        if c3 < 0:
            a2 = -a2
            a4 = -a4
            a6 = -a6
            a8 = -a8
            a10 = -a10
            a12 = -a12
        
        # P(y) = a2 + a4*y + a6*y^2 + a8*y^3 + a10*y^4 + a12*y^5
        coeffs_P = np.array([a2, a4, a6, a8, a10, a12])
        
        # Find real positive roots
        roots = np.roots(coeffs_P)
        real_roots = []
        for root in roots:
            if abs(root.imag) < 1e-9:
                real_root = root.real
                if real_root > 1e-10:
                    real_roots.append(real_root)
        
        if not real_roots:
            return float('inf')
        
        real_roots.sort()
        
        # Vectorized sign-change check
        # For each root y, check if P(y-eps) and P(y+eps) have opposite signs
        eps = 1e-7 * np.maximum(1.0, np.abs(np.array(real_roots)))
        y_arr = np.array(real_roots)
        
        # Evaluate P at y-eps and y+eps using Horner's method
        # P(y) = a2 + a4*y + a6*y^2 + a8*y^3 + a10*y^4 + a12*y^5
        def _eval_P(y_vals):
            # Horner: ((((a12*y + a10)*y + a8)*y + a6)*y + a4)*y + a2
            result = np.full_like(y_vals, a12, dtype=float)
            result = result * y_vals + a10
            result = result * y_vals + a8
            result = result * y_vals + a6
            result = result * y_vals + a4
            result = result * y_vals + a2
            return result
        
        P_minus = _eval_P(y_arr - eps)
        P_plus = _eval_P(y_arr + eps)
        
        # Sign change: product < 0
        sign_changes = P_minus * P_plus < 0
        
        if not np.any(sign_changes):
            return float('inf')
        
        # Largest y with sign change
        best_y = np.max(y_arr[sign_changes])
        
        value = best_y / (2 * math.pi)
        return value
    
    except Exception:
        return float('inf')


def _smooth_surrogate(c0, c1, c2, penalty_weight=100.0, soft_beta=50.0):
    """
    Smooth surrogate objective.
    
    The true objective is: min over valid configs of (largest sign-changing root y) / (2*pi).
    
    This surrogate:
    1. Computes all positive real roots of P(y)
    2. For roots where sign changes, use the largest one (smoothly approximated)
    3. For roots where sign doesn't change, add a penalty to discourage them
    4. Uses soft-max to smoothly select the "largest" sign-changing root
    
    This makes the objective differentiable almost everywhere, enabling gradient-based methods.
    """
    try:
        c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
        
        a2 = c1*(-48) + c2*(-40320) + c3*(-20321280)
        a4 = c1*16 + c2*24192 + c3*20901120
        a6 = c2*(-4480) + c3*(-9067200)
        a8 = c2*256 + c3*1684608
        a10 = c3*(-147456)
        a12 = c3*4096
        
        if c3 < 0:
            a2 = -a2
            a4 = -a4
            a6 = -a6
            a8 = -a8
            a10 = -a10
            a12 = -a12
        
        coeffs_P = np.array([a2, a4, a6, a8, a10, a12])
        
        # Find roots
        roots = np.roots(coeffs_P)
        
        # Filter to real positive roots
        real_roots = []
        for root in roots:
            if abs(root.imag) < 1e-8:
                real_root = root.real
                if real_root > 1e-8:
                    real_roots.append(real_root)
        
        if len(real_roots) == 0:
            return 1e6  # Large penalty for no roots
        
        y_arr = np.array(real_roots)
        
        # Check sign change at each root
        eps = 1e-6 * np.maximum(1.0, np.abs(y_arr))
        
        def _eval_P(y_vals):
            result = np.full_like(y_vals, a12, dtype=float)
            result = result * y_vals + a10
            result = result * y_vals + a8
            result = result * y_vals + a6
            result = result * y_vals + a4
            result = result * y_vals + a2
            return result
        
        P_minus = _eval_P(y_arr - eps)
        P_plus = _eval_P(y_arr + eps)
        
        # Sign change indicator: 1 if sign changes, 0 otherwise
        # Smooth approximation: sigmoid of -|P_minus * P_plus| scaled
        product = P_minus * P_plus
        # If product < 0, sign changes. Use a smooth indicator.
        # soft_sign_change = 1 / (1 + exp(beta * |product| / scale))
        # But we want 1 when product < 0, 0 when product > 0
        # Use: soft_indicator = 1 / (1 + exp(beta * product / scale))
        # When product < 0, exp(beta*product/scale) < 1, so indicator > 0.5
        # When product > 0, exp(beta*product/scale) > 1, so indicator < 0.5
        # This is not quite right. Let's use:
        # soft_indicator = 1 / (1 + exp(-beta * (-sign(product)) * |product| / scale))
        # Actually simpler: we want a smooth function that is ~1 when product<0 and ~0 when product>0
        # Use: soft_ind = 1 / (1 + exp(beta * product / scale))
        # When product is large negative: exp(beta*product/scale) ~ 0, soft_ind ~ 1
        # When product is large positive: exp(beta*product/scale) ~ inf, soft_ind ~ 0
        # When product ~ 0: soft_ind ~ 0.5
        
        scale = np.maximum(1e-10, np.abs(product))
        # Use a smoothed version
        soft_ind = 1.0 / (1.0 + np.exp(soft_beta * product / (scale + 1e-10)))
        
        # Weighted sum: we want the largest y where soft_ind is high
        # Use soft-max: exp(beta * y * soft_ind) / sum(exp(beta * y * soft_ind))
        # Then expected y = sum(weighted_y)
        
        # Combine: objective = (soft_max_y) / (2*pi) + penalty * sum(1 - soft_ind) for non-sign-changing roots
        # Actually, let's just use the largest root weighted by soft_ind
        
        # Soft max selection
        beta_sm = 20.0
        weights = np.exp(beta_sm * y_arr * soft_ind)
        weights = weights / np.sum(weights)
        selected_y = np.sum(weights * y_arr)
        
        # Penalty for roots that don't change sign (to encourage valid configurations)
        # But actually, we just want the largest sign-changing root.
        # The soft_ind already downweights non-sign-changing roots.
        
        value = selected_y / (2 * math.pi)
        
        # Add a small penalty if there are very few sign-changing roots
        # This encourages configurations with at least one clear sign change
        n_sign_changes = np.sum(soft_ind > 0.8)
        if n_sign_changes == 0:
            value += penalty_weight
        
        return value
    
    except Exception:
        return 1e6


def _seed_value():
    """Return the value for the seed coefficients."""
    return _compute_value(-113.0/100.0, 1.0/25.0, 1.0/3240.0)


def construct():
    start_time = time.time()
    
    # Start with the seed
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_value = _compute_value(best_c[0], best_c[1], best_c[2])
    
    def _time_left():
        return _BUDGET_SECONDS - (time.time() - start_time)
    
    # Strategy 1: Local optimization from seed using Nelder-Mead on the exact objective
    if _time_left() > 40:
        try:
            res = minimize(
                lambda x: _compute_value(x[0], x[1], x[2]),
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Strategy 2: Differential evolution with wider bounds
    if _time_left() > 80:
        try:
            bounds = [
                (-10.0, 10.0),
                (-10.0, 10.0),
                (-1.0, 1.0)
            ]
            res = differential_evolution(
                lambda x: _compute_value(x[0], x[1], x[2]),
                bounds,
                maxiter=150,
                popsize=40,
                tol=1e-12,
                seed=42,
                init='sobol'
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Strategy 3: Multi-start local optimization using the smooth surrogate first, then exact
    if _time_left() > 60:
        rng = np.random.default_rng(12345)
        n_starts = 0
        while _time_left() > 25:
            x0 = rng.uniform(-8.0, 8.0, size=3)
            x0[2] *= 0.2  # c2 is typically small
            
            # First, optimize the smooth surrogate to get a good starting point
            try:
                res_smooth = minimize(
                    lambda x: _smooth_surrogate(x[0], x[1], x[2]),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                
                # Then polish with the exact objective
                res_exact = minimize(
                    lambda x: _compute_value(x[0], x[1], x[2]),
                    res_smooth.x,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15}
                )
                
                if res_exact.fun < best_value:
                    best_value = res_exact.fun
                    best_c = res_exact.x
            except Exception:
                pass
            
            n_starts += 1
            if n_starts > 150:
                break
    
    # Strategy 4: Final high-precision polish pass
    if _time_left() > 15:
        try:
            res = minimize(
                lambda x: _compute_value(x[0], x[1], x[2]),
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-17}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Strategy 5: Try a few more random restarts if time permits
    if _time_left() > 20:
        rng2 = np.random.default_rng(99999)
        for _ in range(20):
            if _time_left() < 10:
                break
            x0 = rng2.uniform(-6.0, 6.0, size=3)
            x0[2] *= 0.15
            try:
                res = minimize(
                    lambda x: _compute_value(x[0], x[1], x[2]),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 4000, 'xatol': 1e-13, 'fatol': 1e-15}
                )
                if res.fun < best_value:
                    best_value = res.fun
                    best_c = res.x
            except Exception:
                pass
    
    return np.array(best_c, dtype=np.float64)

# EVOLVE-BLOCK-END
