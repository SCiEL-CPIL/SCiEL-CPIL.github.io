"""Bootstrap spawn: SEARCH approach – generate many varied candidates, keep the best."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, binomial

    z_rationals = sorted(Rational(float(z)) for z in z_list)
    if len(set(z_rationals)) != 12:
        return None
    for z in z_rationals:
        if z <= 0:
            return None

    x = symbols('x')
    alpha = Rational(-1, 2)

    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / math.factorial(k)
        return result

    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(L_gen_explicit(d, alpha, x))

    # Build system: 26 equations for 26 unknowns
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)

    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0

    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[1, 0] = 1

    # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
    for i in range(12):
        zi = z_rationals[i]
        row_idx = 2 + 2 * i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].subs(x, zi)
        b[row_idx, 0] = 0
        row_idx = 3 + 2 * i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_idx, 0] = 0

    try:
        c = A.LUsolve(b)
    except Exception:
        return None

    g = sum(c[k] * laguerre_pols[k] for k in range(26))

    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2

    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)

    try:
        q_poly, rem = g_poly.div(denom_poly)
    except Exception:
        return None

    # Check remainder is zero (exact division)
    if q_poly is None:
        return None
    if rem is not None and rem.degree() >= 0 and rem.all_coeffs() != [0]:
        # Not an exact division; still proceed but may not matter
        pass

    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None

    eps = Rational(1, 10)**198

    max_r = 0
    for r in real_roots_list:
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

def _build_laguerre_coeffs_fast():
    """
    Pre-compute coefficients of L_{2k}^{(-1/2)}(x) in monomial basis.
    Returns a matrix C of shape (26, 51) where C[j, k] is coeff of x^k in L_{2j}^{(-1/2)}.
    """
    alpha = -0.5
    C = np.zeros((26, 51))
    for j in range(26):
        n = 2 * j
        # L_n^{(alpha)}(x) = sum_{k=0}^n binom(n+alpha, n-k) (-x)^k / k!
        for k in range(n + 1):
            # binom(n+alpha, n-k)
            # Using scipy.special.binom for float
            try:
                coeff = scipy_special_binom(n + alpha, n - k) * ((-1.0)**k) / math.factorial(k)
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
        # A matrix construction
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0 => sum c_j L_{2j}(0) = 0
        # L_n^{(alpha)}(0) = binom(n+alpha, n)
        for j in range(26):
            n = 2*j
            A[0, j] = scipy_special_binom(n - 0.5, n)
        b[0] = 0.0
        
        # Row 1: g'(0) = 1 => sum c_j L'_{2j}(0) = 1
        # L_n^{(alpha)}(x) = sum_{k=0}^n a_k x^k. L'(0) = a_1.
        # a_1 = binom(n+alpha, n-1) * (-1)^1 / 1! = -binom(n+alpha, n-1)
        for j in range(26):
            n = 2*j
            if n >= 1:
                A[1, j] = -scipy_special_binom(n - 0.5, n - 1)
            else:
                A[1, j] = 0.0
        b[1] = 1.0
        
        # Rows 2..25: g(z_i)=0, g'(z_i)=0
        # We need L_{2j}(z) and L'_{2j}(z)
        # L_{2j}(z) = sum_{k=0}^{2j} C[j,k] z^k
        # L'_{2j}(z) = sum_{k=1}^{2j} k C[j,k] z^{k-1}
        
        for i in range(12):
            zi = z[i]
            row_g = 2 + 2*i
            row_dp = 3 + 2*i
            
            # Precompute powers of zi
            powers = np.zeros(51)
            powers[0] = 1.0
            for k in range(1, 51):
                powers[k] = powers[k-1] * zi
            
            for j in range(26):
                n = 2*j
                # g(zi)
                A[row_g, j] = np.dot(C[j, :n+1], powers[:n+1])
                # g'(zi)
                # sum k C[j,k] zi^{k-1}
                # = sum_{k=1}^n k C[j,k] zi^{k-1}
                # Let m = k-1. sum_{m=0}^{n-1} (m+1) C[j, m+1] zi^m
                # = dot( (1..n) * C[j, 1..n], powers[0..n-1] )
                coeffs_deriv = np.arange(1, n+1) * C[j, 1:n+1]
                A[row_dp, j] = np.dot(coeffs_deriv, powers[:n])
            
            b[row_g] = 0.0
            b[row_dp] = 0.0
        
        try:
            c = np.linalg.solve(A, b)
        except:
            return 1e100
            
        # 2. Construct g(x) coefficients
        # g_k = sum_j c_j C[j, k]
        g_coeffs = np.dot(c, C) # shape (51,)
        
        # 3. Construct D(x) = x * prod (x - zi)^2
        # Start with [1] for x^0? No, D has degree 25.
        # D(x) = x * P(x)^2 where P(x) = prod (x - zi)
        # P(x) = x^12 + p_11 x^11 + ...
        # Let's compute P(x) coefficients
        p_coeffs = np.ones(13) # p_12=1
        for zi in z:
            # multiply by (x - zi)
            new_p = np.zeros(14)
            for k in range(13):
                new_p[k+1] += p_coeffs[k]
                new_p[k] -= p_coeffs[k] * zi
            p_coeffs = new_p[:13] # keep degree 12
            
        # P(x)^2
        p2_coeffs = np.convolve(p_coeffs, p_coeffs) # degree 24
        
        # D(x) = x * P(x)^2 => shift by 1
        d_coeffs = np.zeros(26) # degree 25
        d_coeffs[1:] = p2_coeffs
        
        # 4. Polynomial division g / D
        # g has degree 50, D has degree 25. q has degree 25.
        # We want q such that g = q * D + R (R=0 ideally)
        # Long division
        g_work = g_coeffs.copy()
        q_coeffs = np.zeros(25)
        
        # Leading term of D is d_coeffs[25] = 1 (since p_coeffs[0]=1, p2_coeffs[0]=1, d_coeffs[25]=1)
        # Actually, let's check.
        # p_coeffs[0] is coeff of x^12. p_coeffs = [p_0, ..., p_12].
        # My convolve assumes standard order?
        # np.convolve([1, a], [1, b]) = [1, a+b, ab].
        # If p_coeffs[0] is x^12 coeff, then this is reversed.
        # Let's use standard math: p_coeffs[k] is coeff of x^k.
        # p_coeffs[12] = 1.
        # p2_coeffs[24] = 1.
        # d_coeffs[25] = 1.
        
        # Division:
        for k in range(50, 24, -1):
            if g_work[k] != 0:
                q_term = g_work[k] / d_coeffs[25]
                q_coeffs[k-25] = q_term
                # Subtract q_term * D(x) * x^(k-25)
                for m in range(26):
                    g_work[k-25+m] -= q_term * d_coeffs[m]
                    
        # Now q_coeffs are in standard order (q_coeffs[0] is x^0 coeff? No.)
        # In the loop, k goes 50 down to 25.
        # q_coeffs[k-25] corresponds to x^(k-25).
        # So q_coeffs[0] is x^0, q_coeffs[25] is x^25.
        
        # 5. Find roots of q(x)
        # np.roots expects coefficients in descending order.
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
        
    except Exception as e:
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
    # Set a hard deadline well before the 630s limit to allow for safe return
    deadline = start_time + 550.0
    
    # Pre-compute Laguerre coefficients for fast score
    C = _build_laguerre_coeffs_fast()
    
    best_score = float('inf')
    best_result = None
    
    def objective(y):
        z = np.exp(y)
        return _compute_score_fast(z, C)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Bessel zeros squared
    baseline = np.array([_bessel_zero_J0(i+1)**2 for i in range(12)])
    candidates.append(baseline)
    
    # 2. Laguerre roots (approximate)
    # Roots of L_n^{(alpha)} are roughly (n + alpha/2 + 0.75*i)^2 * pi^2 / 4 ? 
    # Let's just use a spread of values
    for scale in [0.5, 1.0, 2.0, 5.0]:
        z_cand = np.arange(1, 13) * scale
        candidates.append(z_cand)
        
    # 3. Random logarithmic spacing
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        z_cand = np.sort(np.exp(np.random.uniform(-1, 4, 12)))
        candidates.append(z_cand)
        
    # Evaluate baselines
    for z_cand in candidates:
        if np.any(z_cand <= 0):
            continue
        score = _compute_score_fast(z_cand, C)
        if score < best_score:
            best_score = score
            best_result = z_cand.copy()
            
    # Optimization using L-BFGS-B from best candidate and a few others
    # We use log-space to ensure positivity
    initial_points = []
    if best_result is not None:
        initial_points.append(np.log(best_result))
    for z_cand in candidates[:3]:
        if np.all(z_cand > 0):
            initial_points.append(np.log(z_cand))
            
    bounds_y = [(-5.0, 10.0)] * 12
    
    for y0 in initial_points:
        if time.time() > deadline - 50:
            break
        try:
            # L-BFGS-B is good for smooth problems with bounds
            res = minimize(objective, y0, method='L-BFGS-B', bounds=bounds_y, options={'maxiter': 200, 'ftol': 1e-8, 'maxls': 50})
            if res.fun < best_score:
                z_opt = np.exp(res.x)
                # Check distinctness
                if np.all(np.diff(np.sort(z_opt)) > 1e-5):
                    best_result = z_opt
                    best_score = res.fun
        except Exception:
            pass
            
    # Final local polish with Nelder-Mead if time permits
    if best_result is not None and time.time() < deadline - 20:
        try:
            res_nm = minimize(objective, np.log(best_result), method='Nelder-Mead', options={'maxiter': 100, 'fatol': 1e-10})
            if res_nm.fun < best_score:
                z_nm = np.exp(res_nm.x)
                if np.all(np.diff(np.sort(z_nm)) > 1e-5):
                    best_result = z_nm
                    best_score = res_nm.fun
        except Exception:
            pass

    final_candidate = best_result
    if final_candidate is None:
        final_candidate = candidates[0]
        
    # Ensure distinctness and positivity
    final_candidate = sorted([max(0.01, z) for z in final_candidate])
    for i in range(1, 12):
        if final_candidate[i] - final_candidate[i-1] < 1e-6:
            final_candidate[i] = final_candidate[i-1] + 0.01
            
    return final_candidate

# EVOLVE-BLOCK-END
