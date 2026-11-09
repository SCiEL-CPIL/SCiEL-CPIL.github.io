"""Refined program for Laguerre double roots (m=12, n=1)."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _laguerre_coeffs(n, alpha):
    """
    Return the coefficients of the generalized Laguerre polynomial L_n^(alpha)(x)
    as a numpy array [c_n, c_{n-1}, ..., c_0].
    """
    # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
    # Coeff of x^k is (-1)^k * binom(n+alpha, n-k) / k!
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        # binom(n+alpha, n-k)
        # n+alpha is not integer if alpha is not integer.
        # Use gamma function or product.
        # binom(a, b) = Gamma(a+1) / (Gamma(b+1) Gamma(a-b+1))
        # Here a = n+alpha, b = n-k.
        # a - b + 1 = n+alpha - (n-k) + 1 = k + alpha + 1
        # So term is Gamma(n+alpha+1) / (Gamma(n-k+1) Gamma(k+alpha+1)) * (-1)^k / k!
        # This might be numerically unstable for large n.
        # Alternative: Recurrence for coefficients.
        pass
    
    # Let's use the recurrence for the polynomial itself to get coefficients.
    # L_0 = 1
    # L_1 = 1+alpha - x
    # (k+1) L_{k+1} = (2k+alpha+1-x)L_k - (k+alpha)L_{k-1}
    
    if n == 0:
        return np.array([1.0])
    
    L_prev = np.array([1.0])
    L_curr = np.array([1.0 + alpha, -1.0]) # Coeffs for x^1, x^0 -> [ -1, 1+alpha ]?
    # np.poly1d uses [highest, ..., lowest]
    # L_1(x) = -x + (1+alpha). Coeffs: [-1, 1+alpha]
    
    if n == 1:
        return L_curr
    
    for k in range(1, n):
        # L_{k+1} = ((2k+alpha+1-x)L_k - (k+alpha)L_{k-1}) / (k+1)
        # (2k+alpha+1-x)L_k is a polynomial multiplication.
        # L_k has degree k. L_{k-1} has degree k-1.
        
        # Pad L_curr to length k+2
        L_curr_padded = np.zeros(k + 2)
        L_curr_padded[:len(L_curr)] = L_curr
        
        # Multiply by (2k+alpha+1 - x)
        # Coeffs of (2k+alpha+1 - x) are [-1, 2k+alpha+1]
        # Convolution
        term1 = np.convolve(L_curr, [-1.0, 2.0 * k + alpha + 1.0])
        
        # Multiply L_prev by (k+alpha)
        term2 = (k + alpha) * L_prev
        
        # Align lengths
        max_len = max(len(term1), len(term2))
        term1_padded = np.zeros(max_len)
        term2_padded = np.zeros(max_len)
        term1_padded[:len(term1)] = term1
        term2_padded[:len(term2)] = term2
        
        L_next = (term1_padded - term2_padded) / (k + 1.0)
        
        L_prev = L_curr
        L_curr = L_next
        
    return L_curr

def _compute_score_float(z_list):
    """
    Fast floating point approximation of the score.
    """
    z = np.array(z_list)
    if len(z) != 12:
        return 1e10
    if np.any(z <= 0):
        return 1e10
    if np.any(np.diff(np.sort(z)) < 1e-12):
        return 1e10
    
    alpha = -0.5
    
    # Precompute Laguerre polynomials L_{2k}^{(-1/2)}(x) and their derivatives
    # We need values at z_i and 0.
    # L_{2k}(0) and L'_{2k}(0)
    # L_{2k}(z_i) and L'_{2k}(z_i)
    
    # Recurrence for L_n and L'_n
    # L'_n^{(alpha)}(x) = -L_{n-1}^{(alpha+1)}(x)
    # So we need L_n^{(-1/2)} and L_n^{(1/2)}.
    
    # We need L_{2k} for k=0..25. So n up to 50.
    
    # Compute L_n^{(-1/2)} and L_n^{(1/2)} for n=0..50 at all points
    points = np.concatenate(([0.0], z))
    N_pts = len(points)
    
    # L_alpha1 for alpha=-0.5 is L_n^{(0.5)}
    # L_alpha for alpha=-0.5 is L_n^{(-0.5)}
    
    # Recurrence:
    # L_0 = 1
    # L_1 = 1+alpha-x
    # L_{k+1} = ((2k+alpha+1-x)L_k - (k+alpha)L_{k-1})/(k+1)
    
    L_minus = np.zeros((51, N_pts))
    L_plus = np.zeros((51, N_pts))
    
    # Init
    L_minus[0, :] = 1.0
    L_plus[0, :] = 1.0
    L_minus[1, :] = 1.0 + (-0.5) - points
    L_plus[1, :] = 1.0 + 0.5 - points
    
    for k in range(1, 50):
        # For L_minus (alpha = -0.5)
        L_minus[k+1, :] = ((2.0 * k - 0.5 + 1.0 - points) * L_minus[k, :] - (k - 0.5) * L_minus[k-1, :]) / (k + 1.0)
        # For L_plus (alpha = 0.5)
        L_plus[k+1, :] = ((2.0 * k + 0.5 + 1.0 - points) * L_plus[k, :] - (k + 0.5) * L_plus[k-1, :]) / (k + 1.0)
        
    # Derivatives of L_minus: L'_n^{(-0.5)} = -L_{n-1}^{(0.5)}
    # So L'_{2k} = -L_{2k-1}^{(0.5)}
    
    # Build Matrix A (26x26)
    # Rows 0,1: at x=0
    # Rows 2..25: at x=z_i
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # k=0..25 corresponds to L_{2k}
    for k in range(26):
        n = 2 * k
        # Value at 0
        A[0, k] = L_minus[n, 0]
        # Derivative at 0: -L_{n-1}^{(0.5)}(0). If n=0, deriv is 0.
        if n == 0:
            A[1, k] = 0.0
        else:
            A[1, k] = -L_plus[n-1, 0]
            
        # Values at z_i
        for i in range(12):
            idx = i + 1
            pt_idx = 1 + i
            A[2 + 2*i, k] = L_minus[n, pt_idx]
            if n == 0:
                A[3 + 2*i, k] = 0.0
            else:
                A[3 + 2*i, k] = -L_plus[n-1, pt_idx]
                
    b[0] = 0.0
    b[1] = 1.0
    
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e10
        
    # Construct g(x) coefficients
    # g(x) = sum c_k L_{2k}(x)
    # We need coefficients of L_{2k}^{(-0.5)}(x)
    g_coeffs = np.zeros(51)
    for k in range(26):
        n = 2 * k
        L_coeffs = _laguerre_coeffs(n, -0.5)
        g_coeffs[:len(L_coeffs)] += c[k] * L_coeffs
        
    # Construct D(x) = x * prod (x-z_i)^2
    D_coeffs = np.array([1.0, 0.0]) # x
    for i in range(12):
        # (x - z_i)^2 = x^2 - 2 z_i x + z_i^2
        quad = np.array([1.0, -2.0 * z[i], z[i]**2])
        D_coeffs = np.convolve(D_coeffs, quad)
        
    # q(x) = g(x) / D(x)
    # g degree 50, D degree 25. q degree 25.
    try:
        q_coeffs, remainder = np.polydiv(g_coeffs, D_coeffs)
    except Exception:
        return 1e10
        
    # Find roots of q
    roots = np.roots(q_coeffs)
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
            
    if not real_roots:
        return 1e10
        
    # Check sign change condition?
    # For float, assume simple roots.
    # Find largest real root
    B = max(real_roots)
    if B < 0:
        B = 0.0
        
    score = B / (2.0 * math.pi)
    return score

def _compute_score_exact(z_list, deadline):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import (symbols, Rational, Poly, diff, real_roots, pi as sympy_pi,
                       Matrix, laguerre as sympy_laguerre)
    
    if time.time() > deadline:
        return None
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    laguerre_pols = []
    for d in range(0, 51, 2):
        try:
            lp = sympy_laguerre(d, alpha, x)
            laguerre_pols.append(lp)
        except Exception:
            return None
    
    if time.time() > deadline:
        return None
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[0, 0] = 0
    b[1, 0] = 1
    
    for i in range(12):
        zi = z_rationals[i]
        row_val = 2 + 2 * i
        row_der = 3 + 2 * i
        for k in range(26):
            A[row_val, k] = laguerre_pols[k].subs(x, zi)
            A[row_der, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_val, 0] = 0
        b[row_der, 0] = 0
    
    if time.time() > deadline:
        return None
    
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, remainder = g_poly.div(denom_poly)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        if time.time() > deadline:
            return None
        try:
            q_at_minus = q_expr.subs(x, r - eps)
            q_at_plus = q_expr.subs(x, r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def _is_valid_z(z_list):
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-12:
                return False
    return True

def construct():
    deadline = time.time() + 560.0
    
    # Initial guess: Bessel zeros squared
    z_bessel = []
    for k in range(1, 13):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            j0 = 2.4048255576957728
        else:
            j0 = x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
        z_bessel.append(j0**2)
    
    if not _is_valid_z(z_bessel):
        # Fallback
        z_bessel = [0.5 * (1.5 ** i) for i in range(12)]
        
    best_z = list(z_bessel)
    best_score_float = _compute_score_float(best_z)
    
    # Optimization using Nelder-Mead on log(z)
    def objective(u):
        z = np.exp(u)
        if not _is_valid_z(z):
            return 1e10
        return _compute_score_float(z)
        
    u0 = np.log(np.array(z_bessel))
    
    # Run minimize
    try:
        res = minimize(objective, u0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success:
            u_opt = res.x
            z_opt = np.exp(u_opt)
            if _is_valid_z(z_opt):
                score_opt = _compute_score_float(z_opt)
                if score_opt < best_score_float:
                    best_z = list(z_opt)
                    best_score_float = score_opt
    except Exception:
        pass
        
    # Exact evaluation of the best candidate
    final_score = None
    if time.time() < deadline - 60:
        final_score = _compute_score_exact(best_z, deadline - 30)
        
    # If time allows, try a few perturbations
    if time.time() < deadline - 90 and final_score is not None:
        # Simple local search
        for i in range(12):
            if time.time() > deadline - 60:
                break
            for step in [0.1, 0.01, 0.5]:
                for sign in [1, -1]:
                    if time.time() > deadline - 30:
                        break
                    cand = list(best_z)
                    cand[i] = best_z[i] * (1 + sign * step)
                    if not _is_valid_z(cand):
                        continue
                    try:
                        s = _compute_score_exact(cand, deadline - 10)
                        if s is not None and s < final_score:
                            final_score = s
                            best_z = cand
                    except Exception:
                        continue
                        
    return best_z

# EVOLVE-BLOCK-END
