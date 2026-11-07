import numpy as np
from scipy.optimize import minimize
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

# Precompute Hermite coefficients
_H0 = np.array(_get_hermite_coeffs(0))
_H4 = np.array(_get_hermite_coeffs(4))
_H8 = np.array(_get_hermite_coeffs(8))
_H12 = np.array(_get_hermite_coeffs(12))

# Values at x=0
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]

def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients."""
    # Build g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    max_deg = 12
    g_coeffs = np.zeros(max_deg + 1)
    
    for coeff, H in [(c0, _H0), (c1, _H4), (c2, _H8)]:
        g_coeffs[:len(H)] += coeff * H
    
    # Compute c3 so that g(0) = 0
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    g_coeffs[:len(_H12)] += c3 * _H12
    
    # Check if g tends to negative at infinity
    if g_coeffs[12] < 0:
        g_coeffs = -g_coeffs
    
    # q = g / x^2
    if abs(g_coeffs[0]) > 1e-12 or abs(g_coeffs[1]) > 1e-12:
        return float('inf')
    
    q_coeffs = g_coeffs[2:]  # Divide by x^2
    
    if len(q_coeffs) == 0:
        return float('inf')
    
    coeffs_desc = list(reversed(q_coeffs))
    roots = np.roots(coeffs_desc)
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
    
    # For each positive real root, check if sign changes
    poly = np.poly1d(coeffs_desc)
    
    largest = None
    for r in real_roots:
        if r <= 0:
            continue
        # Check sign change: evaluate at r - eps and r + eps
        eps = 1e-8 * max(1.0, abs(r))
        val_left = poly(r - eps)
        val_right = poly(r + eps)
        if val_left * val_right < 0:
            # Sign change at r
            if largest is None or r > largest:
                largest = r
    
    if largest is None:
        return float('inf')
    
    return largest ** 2 / (2 * np.pi)

def _compute_value_fast(c0, c1, c2):
    """Faster version using precomputed values where possible."""
    # g(0) = 0 constraint gives c3
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # Build g coefficients
    g = np.zeros(13)
    g[:len(_H0)] += c0 * _H0
    g[:len(_H4)] += c1 * _H4
    g[:len(_H8)] += c2 * _H8
    g[:len(_H12)] += c3 * _H12
    
    # Check leading coefficient
    if g[12] < 0:
        g = -g
    
    # q = g / x^2
    if abs(g[0]) > 1e-12 or abs(g[1]) > 1e-12:
        return float('inf')
    
    q = g[2:]
    coeffs_desc = list(reversed(q))
    roots = np.roots(coeffs_desc)
    
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    if not real_roots:
        return float('inf')
    
    poly = np.poly1d(coeffs_desc)
    largest = None
    for r in real_roots:
        eps = 1e-8 * max(1.0, abs(r))
        if poly(r - eps) * poly(r + eps) < 0:
            if largest is None or r > largest:
                largest = r
    
    if largest is None:
        return float('inf')
    
    return largest ** 2 / (2 * np.pi)

def _smooth_objective(x, beta=20.0):
    """Smooth surrogate using log-sum-exp over sign-changing roots."""
    c0, c1, c2 = x
    try:
        c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
        
        g = np.zeros(13)
        g[:len(_H0)] += c0 * _H0
        g[:len(_H4)] += c1 * _H4
        g[:len(_H8)] += c2 * _H8
        g[:len(_H12)] += c3 * _H12
        
        if g[12] < 0:
            g = -g
        
        if abs(g[0]) > 1e-12 or abs(g[1]) > 1e-12:
            return 1e10
        
        q = g[2:]
        coeffs_desc = list(reversed(q))
        roots = np.roots(coeffs_desc)
        
        sign_changing_costs = []
        poly = np.poly1d(coeffs_desc)
        
        for r in roots:
            if abs(r.imag) < 1e-10 and r.real > 0:
                rr = r.real
                eps = 1e-8 * max(1.0, abs(rr))
                if poly(rr - eps) * poly(rr + eps) < 0:
                    cost = rr ** 2 / (2.0 * np.pi)
                    sign_changing_costs.append(cost)
        
        if not sign_changing_costs:
            return 1e10
        
        costs = np.array(sign_changing_costs)
        # Log-sum-exp for numerical stability
        max_cost = np.max(costs)
        log_sum_exp = max_cost + np.log(np.sum(np.exp(beta * (costs - max_cost))))
        return log_sum_exp / beta
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s
    
    # Seed
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])
    
    def objective(x):
        try:
            return _compute_value_fast(x[0], x[1], x[2])
        except:
            return 1e10
    
    # Phase 1: Multi-start Nelder-Mead for local refinement
    # The landscape is smooth-ish 2-DOF, so local search from good starts is effective
    np.random.seed(42)
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x0 = best_c.copy()
        elif i == 1:
            # Small perturbation around seed
            x0 = best_c + np.random.normal(0, 0.01, size=3)
        elif i == 2:
            # Medium perturbation
            x0 = best_c + np.random.normal(0, 0.1, size=3)
        elif i == 3:
            # Larger perturbation
            x0 = best_c + np.random.normal(0, 0.5, size=3)
        elif i % 3 == 0:
            # Random uniform in reasonable range
            x0 = np.random.uniform(-3, 3, size=3)
        else:
            # Mix of small and medium perturbations
            scale = np.random.choice([0.05, 0.2, 0.3])
            x0 = best_c + np.random.normal(0, scale, size=3)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            val = objective(result.x)
            if val < best_val:
                best_val = val
                best_c = result.x.copy()
        except:
            pass
    
    # Phase 2: Smooth surrogate refinement with log-sum-exp
    # This helps navigate the nonsmooth min/max objective
    if time.time() < deadline - 20:
        for beta in [5.0, 10.0, 20.0, 50.0]:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(lambda x: _smooth_objective(x, beta), best_c,
                                 method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                # Verify with exact objective
                val = objective(result.x)
                if val < best_val:
                    best_val = val
                    best_c = result.x.copy()
            except:
                pass
    
    # Phase 3: High-precision polish with tighter tolerance
    # Multiple rounds of polishing to squeeze out precision
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            try:
                result = minimize(objective, best_c, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
                val = objective(result.x)
                if val < best_val:
                    best_val = val
                    best_c = result.x.copy()
            except:
                pass
    
    # Final verification
    final_val = _compute_value(best_c[0], best_c[1], best_c[2])
    if final_val < best_val:
        best_val = final_val
    
    return best_c
