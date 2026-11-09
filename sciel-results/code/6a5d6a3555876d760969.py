"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre
import time

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 570
    
    def laguerre_polynomial(x, m, n):
        """Compute Laguerre polynomial L_m^(n)(x) using recurrence"""
        if m == 0:
            return np.ones_like(x)
        elif m == 1:
            return -(x + n)
        else:
            L_prev = np.ones_like(x)
            L_curr = -(x + n)
            for k in range(2, m + 1):
                L_new = ((2*k - 1 + n - x) * L_curr - (k - 1 + n) * L_prev) / k
                L_prev, L_curr = L_curr, L_new
            return L_curr
    
    def compute_last_sign_change(roots, m=12, n=1):
        """
        Compute the location of the last sign change in the Laguerre polynomial
        constructed with the given double roots.
        
        The Cohn-Gonçalves bound is related to where the last sign change occurs.
        For double roots at r_1 < r_2 < ... < r_m, the polynomial has sign changes
        at these locations. The bound C4 is proportional to the last sign change location.
        """
        x_max = max(roots) + 5
        x = np.linspace(0, x_max, 2000)
        
        # Compute L_m^(n)(x)
        L_m = laguerre_polynomial(x, m, n)
        
        # Find sign changes
        signs = np.sign(L_m)
        # Replace zeros with previous non-zero sign
        for i in range(1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i-1]
        
        sign_changes = np.where(np.diff(signs) != 0)[0]
        
        if len(sign_changes) == 0:
            return x_max
        
        # Last sign change location
        last_change_idx = sign_changes[-1]
        # Interpolate to find exact location
        if last_change_idx < len(x) - 1:
            # Linear interpolation
            x1, x2 = x[last_change_idx], x[last_change_idx + 1]
            y1, y2 = L_m[last_change_idx], L_m[last_change_idx + 1]
            if y1 != y2:
                t = -y1 / (y2 - y1)
                last_change_x = x1 + t * (x2 - x1)
            else:
                last_change_x = (x1 + x2) / 2
        else:
            last_change_x = x[-1]
        
        return last_change_x
    
    def compute_bound_from_roots(roots):
        """
        Compute the Cohn-Gonçalves upper bound C4 from root locations.
        The bound is related to the last sign change location and the
        weight integral.
        """
        m = 12
        n = 1
        
        # The bound involves the integral of the weight function up to the last sign change
        last_change = compute_last_sign_change(roots, m, n)
        
        # Integrate e^(-x) from 0 to last_change
        # This gives the "mass" that contributes to the bound
        x = np.linspace(0, max(last_change, 1e-10), 1000)
        weight_integral = np.trapz(np.exp(-x), x)
        
        # The total weight integral (for normalization)
        x_total = np.linspace(0, 100, 10000)
        total_weight = np.trapz(np.exp(-x_total), x_total)
        
        # C4 bound is proportional to this ratio
        bound = weight_integral / total_weight
        
        return bound
    
    # Strategy: Use Laguerre roots as the primary solution
    # The roots of L_12^(1)(x) are known to be optimal for the weighted problem
    try:
        roots_lag, _ = roots_genlaguerre(12, 1)
        x0 = np.sort(roots_lag)
        x0 = np.maximum(x0, 1e-6)
        if np.any(np.diff(x0) <= 1e-6):
            x0 = np.array([4.0 * (i + 1) for i in range(12)])
    except Exception:
        x0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Also try arithmetic progression as fallback
    x_ap = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Evaluate both
    score_lag = compute_bound_from_roots(x0)
    score_ap = compute_bound_from_roots(x_ap)
    
    # Use the better one
    if score_lag < score_ap:
        best_roots = x0
    else:
        best_roots = x_ap
    
    # Try to improve with limited optimization
    def objective(roots):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
            return 1e10
        return compute_bound_from_roots(roots)
    
    # Try SLSQP from the best initial point
    try:
        result = minimize(
            objective,
            best_roots.copy(),
            method='SLSQP',
            bounds=[(1e-6, 100)] * 12,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        
        if result.success and np.all(result.x > 0) and np.all(np.diff(result.x) > 0):
            score_opt = result.fun
            if score_opt < min(score_lag, score_ap):
                best_roots = result.x
    except Exception:
        pass
    
    return list(best_roots)
# EVOLVE-BLOCK-END
