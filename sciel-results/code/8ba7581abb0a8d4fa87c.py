"""Bootstrap spawn: SEARCH approach – generate many varied candidates, keep the best."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import binom as scipy_binom

# EVOLVE-BLOCK-START

def _build_laguerre_coeffs_fast():
    """
    Pre-compute coefficients of L_{2k}^{(-1/2)}(x) in monomial basis.
    Returns a matrix C of shape (26, 51) where C[j, k] is coeff of x^k in L_{2j}^{(-1/2)}.
    """
    alpha = -0.5
    C = np.zeros((26, 51))
    for j in range(26):
        n = 2 * j
        for k in range(n + 1):
            try:
                coeff = scipy_binom(n + alpha, n - k) * ((-1.0) ** k) / math.factorial(k)
            except:
                coeff = 0.0
            C[j, k] = coeff
    return C

def _compute_score_fast(z_list, C):
    """
    Fast numerical approximation of the score.
    z_list: list of 12 floats.
    C: pre-computed Laguerre coefficient matrix.
    """
    try:
        z = np.array(z_list, dtype=float)
        if np.any(z <= 0) or np.any(np.diff(np.sort(z)) < 1e-8):
            return 1e100
        
        # 1. Solve for c
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        for j in range(26):
            n = 2*j
            A[0, j] = scipy_binom(n - 0.5, n)
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        for j in range(26):
            n = 2*j
            if n >= 1:
                A[1, j] = -scipy_binom(n - 0.5, n - 1)
            else:
                A[1, j] = 0.0
        b[1] = 1.0
        
        # Rows 2..25: g(z_i)=0, g'(z_i)=0
        for i in range(12):
            zi = z[i]
            row_g = 2 + 2*i
            row_dp = 3 + 2*i
            
            powers = np.zeros(51)
            powers[0] = 1.0
            for k in range(1, 51):
                powers[k] = powers[k-1] * zi
            
            for j in range(26):
                n = 2*j
                A[row_g, j] = np.dot(C[j, :n+1], powers[:n+1])
                coeffs_deriv = np.arange(1, n+1) * C[j, 1:n+1]
                A[row_dp, j] = np.dot(coeffs_deriv, powers[:n])
            
            b[row_g] = 0.0
            b[row_dp] = 0.0
        
        try:
            c = np.linalg.solve(A, b)
        except:
            return 1e100
            
        # 2. Construct g(x) coefficients
        g_coeffs = np.dot(c, C) # shape (51,)
        
        # 3. Construct D(x) = x * prod (x - zi)^2
        p_coeffs = np.ones(13)
        for zi in z:
            new_p = np.zeros(14)
            for k in range(13):
                new_p[k+1] += p_coeffs[k]
                new_p[k] -= p_coeffs[k] * zi
            p_coeffs = new_p[:13]
            
        p2_coeffs = np.convolve(p_coeffs, p_coeffs)
        d_coeffs = np.zeros(26)
        d_coeffs[1:] = p2_coeffs
        
        # 4. Polynomial division g / D
        g_work = g_coeffs.copy()
        q_coeffs = np.zeros(25)
        
        for k in range(50, 24, -1):
            if g_work[k] != 0:
                q_term = g_work[k] / d_coeffs[25]
                q_coeffs[k-25] = q_term
                for m in range(26):
                    g_work[k-25+m] -= q_term * d_coeffs[m]
                    
        # 5. Find roots of q(x)
        q_roots = np.roots(q_coeffs[::-1])
        
        # 6. Filter real roots
        real_roots = [r.real for r in q_roots if abs(r.imag) < 1e-9]
        
        if not real_roots:
            return 1e100
            
        # 7. Check sign change
        eps = 1e-9
        max_r = 0
        for r in real_roots:
            q_minus = np.polyval(q_coeffs[::-1], r - eps)
            q_plus = np.polyval(q_coeffs[::-1], r + eps)
            if q_minus * q_plus < 0:
                if r > max_r:
                    max_r = r
                    
        return max_r / (2 * math.pi)
        
    except Exception:
        return 1e100

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def construct():
    """Return 12 double-root locations using SEARCH: many varied candidates, keep best."""
    start_time = time.time()
    deadline = start_time + 550.0
    
    C = _build_laguerre_coeffs_fast()
    
    best_score = float('inf')
    best_result = None
    
    # Initial candidate: Bessel zeros
    baseline = [_bessel_zero_J0(i+1)**2 for i in range(12)]
    best_result = list(baseline)
    best_score = _compute_score_fast(best_result, C)
    
    # Parameterize z_i = exp(y_i) to ensure positivity
    def objective(y):
        z = np.exp(y)
        return _compute_score_fast(z, C)
        
    y0 = np.log(baseline)
    
    # Powell's method
    if time.time() < deadline - 100:
        try:
            res = minimize(objective, y0, method='Powell', options={'maxiter': 50, 'ftol': 1e-4})
            if res.fun < best_score:
                z_opt = np.exp(res.x)
                if np.all(z_opt > 0) and len(set(np.round(z_opt, 10))) == 12:
                    best_result = list(z_opt)
                    best_score = res.fun
        except Exception:
            pass
        
    # Differential Evolution
    bounds = [(-2.0, 5.0)] * 12
    if time.time() < deadline - 100:
        try:
            res_de = differential_evolution(objective, bounds, seed=42, maxiter=10, popsize=5, tol=0.1, polish=False)
            if res_de.fun < best_score:
                z_de = np.exp(res_de.x)
                if np.all(z_de > 0) and len(set(np.round(z_de, 10))) == 12:
                    best_result = list(z_de)
                    best_score = res_de.fun
        except Exception:
            pass

    # Local refinement
    if best_result is not None and time.time() < deadline - 50:
        try:
            res_nm = minimize(objective, np.log(best_result), method='Nelder-Mead', options={'maxiter': 50, 'fatol': 1e-6})
            if res_nm.fun < best_score:
                z_nm = np.exp(res_nm.x)
                if np.all(z_nm > 0) and len(set(np.round(z_nm, 10))) == 12:
                    best_result = list(z_nm)
                    best_score = res_nm.fun
        except Exception:
            pass

    final_candidate = best_result
    if final_candidate is None:
        final_candidate = baseline
        
    # Ensure distinctness and positivity
    final_candidate = sorted([max(0.01, z) for z in final_candidate])
    for i in range(1, 12):
        if final_candidate[i] - final_candidate[i-1] < 1e-6:
            final_candidate[i] = final_candidate[i-1] + 0.01
            
    return final_candidate

# EVOLVE-BLOCK-END
