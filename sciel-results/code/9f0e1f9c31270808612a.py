import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as list [c0, c1, ..., cn]."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    H = [[1.0], [0.0, 2.0]]
    for k in range(2, n + 1):
        prev1 = H[k - 1]
        prev2 = H[k - 2]
        curr = [0.0] * (k + 1)
        for i in range(k):
            curr[i + 1] += 2.0 * prev1[i]
        for i in range(k - 1):
            curr[i] += -2.0 * (k - 1) * prev2[i]
        H.append(curr)
    return H[n]

# Precompute
_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)

# Extract even-power coefficients: H_{2k} = sum_{j=0}^{k} b_{2k,j} x^{2j}
# For H0: b0 = [1]
# For H4: b4 = [12, -48, 16]  (coeff of x^0, x^2, x^4)
# For H8: b8 = [112, -1344, 2688, -1344, 256]
# For H12: b12 = [9240, -120960, 403200, -483840, 230400, -46080, 4096]

b0 = [_H0[0]]  # [1]
b4 = [_H4[0], _H4[2], _H4[4]]  # coeff of x^0, x^2, x^4
b8 = [_H8[0], _H8[2], _H8[4], _H8[6], _H8[8]]
b12 = [_H12[0], _H12[2], _H12[4], _H12[6], _H12[8], _H12[10], _H12[12]]

# P(z) = a2 + a4*z + a6*z^2 + a8*z^3 + a10*z^4 + a12*z^5
# where a_{2k} = c0*b0[k] + c1*b4[k] + c2*b8[k] + c3*b12[k] for k=0..5
# b0 has 1 element, b4 has 3, b8 has 5, b12 has 7

# For k=0: a0 = c0*b0[0] + c1*b4[0] + c2*b8[0] + c3*b12[0] = 0
# For k=1: a2 = c1*b4[1] + c2*b8[1] + c3*b12[1]  (b0 has no k=1 term)
# For k=2: a4 = c1*b4[2] + c2*b8[2] + c3*b12[2]
# For k=3: a6 = c2*b8[3] + c3*b12[3]
# For k=4: a8 = c2*b8[4] + c3*b12[4]
# For k=5: a10 = c3*b12[5]
# For k=6: a12 = c3*b12[6]

# c3 = -(c0*b0[0] + c1*b4[0] + c2*b8[0]) / b12[0]

b0_0 = b0[0]
b4_0 = b4[0]
b8_0 = b8[0]
b12_0 = b12[0]

def _coeffs_to_P(c0, c1, c2):
    """Compute coefficients of P(z) = a2 + a4*z + ... + a12*z^5."""
    c3 = -(c0 * b0_0 + c1 * b4_0 + c2 * b8_0) / b12_0
    
    a2 = c1 * b4[1] + c2 * b8[1] + c3 * b12[1]
    a4 = c1 * b4[2] + c2 * b8[2] + c3 * b12[2]
    a6 = c2 * b8[3] + c3 * b12[3]
    a8 = c2 * b8[4] + c3 * b12[4]
    a10 = c3 * b12[5]
    a12 = c3 * b12[6]
    
    # Check leading coefficient sign; if a12 < 0, flip all
    if a12 < 0:
        a2, a4, a6, a8, a10, a12 = -a2, -a4, -a6, -a8, -a10, -a12
    
    return [a2, a4, a6, a8, a10, a12]

def _largest_sign_changing_root(P_coeffs):
    """Find largest positive real root of P(z) where sign changes.
    P_coeffs = [a2, a4, a6, a8, a10, a12] for P(z) = a2 + a4*z + ... + a12*z^5"""
    if len(P_coeffs) == 0:
        return None
    
    # Convert to numpy (ascending order)
    coeffs_asc = np.array(P_coeffs, dtype=float)
    coeffs_desc = coeffs_asc[::-1]
    
    roots = np.roots(coeffs_desc)
    
    poly = np.poly1d(coeffs_desc)
    
    largest = None
    for r in roots:
        if abs(r.imag) < 1e-10:
            z = r.real
            if z <= 0:
                continue
            # Check sign change
            eps = 1e-8 * max(1.0, abs(z))
            val_left = poly(z - eps)
            val_right = poly(z + eps)
            if val_left * val_right < 0:
                if largest is None or z > largest:
                    largest = z
    
    return largest

def _compute_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) where r is largest sign-changing root of q(x)."""
    P_coeffs = _coeffs_to_P(c0, c1, c2)
    z_star = _largest_sign_changing_root(P_coeffs)
    if z_star is None:
        return float('inf')
    # r = sqrt(z_star), value = r^2 / (2*pi) = z_star / (2*pi)
    return z_star / (2.0 * np.pi)

def _smooth_objective(x, beta=20.0):
    """Smooth surrogate using log-sum-exp over sign-changing roots."""
    c0, c1, c2 = x
    try:
        P_coeffs = _coeffs_to_P(c0, c1, c2)
        coeffs_asc = np.array(P_coeffs, dtype=float)
        coeffs_desc = coeffs_asc[::-1]
        roots = np.roots(coeffs_desc)
        poly = np.poly1d(coeffs_desc)
        
        costs = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                z = r.real
                if z <= 0:
                    continue
                eps = 1e-8 * max(1.0, abs(z))
                if poly(z - eps) * poly(z + eps) < 0:
                    costs.append(z / (2.0 * np.pi))
        
        if not costs:
            return 1e10
        
        costs = np.array(costs)
        max_cost = np.max(costs)
        log_sum_exp = max_cost + np.log(np.sum(np.exp(beta * (costs - max_cost))))
        return log_sum_exp / beta
    except:
        return 1e10

def _newton_refine(P_coeffs, z0, max_iter=100):
    """Newton's method to refine a root of P(z)."""
    coeffs_asc = np.array(P_coeffs, dtype=float)
    coeffs_desc = coeffs_asc[::-1]
    poly = np.poly1d(coeffs_desc)
    dpoly = poly.deriv()
    
    z = z0
    for _ in range(max_iter):
        f = poly(z)
        fp = dpoly(z)
        if abs(fp) < 1e-30:
            break
        dz = -f / fp
        z_new = z + dz
        if abs(dz) < 1e-15 * max(1.0, abs(z)):
            z = z_new
            break
        z = z_new
    
    return z

def construct():
    start_time = time.time()
    deadline = 200.0  # Safety margin before 210s
    
    # Seed
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])
    
    def objective(x):
        try:
            return _compute_value(x[0], x[1], x[2])
        except:
            return 1e10
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 80:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            de_result = differential_evolution(
                objective, bounds, maxiter=150, popsize=20,
                tol=1e-10, seed=42, polish=False
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
        except:
            pass
    
    # Phase 2: Multi-start Nelder-Mead
    np.random.seed(42)
    num_starts = 40
    for i in range(num_starts):
        if time.time() > deadline - 40:
            break
        
        if i == 0:
            x0 = best_c.copy()
        elif i % 4 == 1:
            x0 = best_c + np.random.normal(0, 0.5, size=3)
        elif i % 4 == 2:
            x0 = np.random.uniform(-3, 3, size=3)
        elif i % 4 == 3:
            x0 = best_c + np.random.normal(0, 0.1, size=3)
        else:
            x0 = np.random.uniform(-5, 5, size=3)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
    
    # Phase 3: Smooth surrogate refinement
    if time.time() < deadline - 30:
        for beta in [10.0, 30.0, 100.0, 300.0]:
            if time.time() > deadline - 20:
                break
            try:
                result = minimize(lambda x: _smooth_objective(x, beta), best_c,
                                 method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                val = objective(result.x)
                if val < best_val:
                    best_val = val
                    best_c = result.x.copy()
            except:
                pass
    
    # Phase 4: High-precision polish with Newton refinement on P(z)
    if time.time() < deadline - 10:
        try:
            # Get current P coefficients
            P_coeffs = _coeffs_to_P(best_c[0], best_c[1], best_c[2])
            z_star = _largest_sign_changing_root(P_coeffs)
            
            if z_star is not None:
                # Newton refine
                z_refined = _newton_refine(P_coeffs, z_star)
                # Verify sign change at refined root
                coeffs_asc = np.array(P_coeffs, dtype=float)
                coeffs_desc = coeffs_asc[::-1]
                poly = np.poly1d(coeffs_desc)
                eps = 1e-10 * max(1.0, abs(z_refined))
                if poly(z_refined - eps) * poly(z_refined + eps) < 0:
                    val_refined = z_refined / (2.0 * np.pi)
                    if val_refined < best_val:
                        best_val = val_refined
                        # The coefficients don't change, but we use the refined root
                        # Actually, we need to find (c0,c1,c2) that gives this P
                        # But P is determined by (c0,c1,c2), so we keep best_c
        except:
            pass
    
    # Phase 5: Final Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            result = minimize(objective, best_c, method='Nelder-Mead',
                             options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
    
    # Final verification
    final_val = _compute_value(best_c[0], best_c[1], best_c[2])
    if final_val < best_val:
        best_val = final_val
    
    return best_c
