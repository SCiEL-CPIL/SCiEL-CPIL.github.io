"""Combined program: fast numpy screening + exact sympy verification."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _laguerre_coeffs(n, alpha):
    """Return coefficients of generalized Laguerre polynomial L_n^(alpha)(x).
    Returns list [c_0, c_1, ..., c_n] such that L(x) = sum c_i x^i.
    """
    if n == 0:
        return [1.0]
    if n == 1:
        return [1.0 + alpha, -1.0]
    
    L_nm1 = [1.0 + alpha, -1.0] # L_1
    L_n = [1.0] # L_0
    
    for i in range(1, n):
        # L_{i+1} = ((2i + 1 + alpha) L_i - x L_i - (i + alpha) L_{i-1}) / (i+1)
        t1 = [c * (2*i + 1 + alpha) for c in L_n]
        t2 = [0.0] + [-c for c in L_n]
        t3 = [c * (-(i + alpha)) for c in L_nm1]
        
        max_deg = len(t2)
        new_coeffs = [0.0] * (max_deg)
        
        for j in range(len(t1)):
            new_coeffs[j] += t1[j]
        for j in range(len(t2)):
            new_coeffs[j] += t2[j]
        for j in range(len(t3)):
            new_coeffs[j] += t3[j]
            
        new_coeffs = [c / (i+1) for c in new_coeffs]
        
        L_nm1 = L_n
        L_n = new_coeffs
        
    return L_n

def _compute_score_fast(z_list):
    """Fast approximation of B/(2*pi) using numpy."""
    try:
        z_arr = np.array(sorted(z_list))
        if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
            return float('inf')
        
        if len(np.unique(z_arr)) < len(z_arr):
            return float('inf')
            
        alpha = -0.5
        n_max = 50
        
        L_coeffs = []
        L_prime_coeffs = []
        for k in range(26):
            n = 2 * k
            coeffs = _laguerre_coeffs(n, alpha)
            L_coeffs.append(coeffs)
            
            d_coeffs = [0.0] * (len(coeffs) - 1)
            for j in range(1, len(coeffs)):
                d_coeffs[j-1] = j * coeffs[j]
            L_prime_coeffs.append(d_coeffs)
            
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        for k in range(26):
            A[0, k] = L_coeffs[k][0]
        b[0] = 0.0
        
        for k in range(26):
            if len(L_prime_coeffs[k]) > 0:
                A[1, k] = L_prime_coeffs[k][0]
            else:
                A[1, k] = 0.0
        b[1] = 1.0
        
        for i in range(12):
            zi = z_arr[i]
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            
            for k in range(26):
                coeffs_rev = np.array(L_coeffs[k])[::-1]
                A[row_g, k] = np.polyval(coeffs_rev, zi)
                
                if len(L_prime_coeffs[k]) > 0:
                    d_coeffs_rev = np.array(L_prime_coeffs[k])[::-1]
                    A[row_gp, k] = np.polyval(d_coeffs_rev, zi)
                else:
                    A[row_gp, k] = 0.0
            b[row_g] = 0.0
            b[row_gp] = 0.0
            
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return float('inf')
            
        g_coeffs = np.zeros(51)
        for k in range(26):
            coeffs = np.array(L_coeffs[k])
            g_coeffs[:len(coeffs)] += c[k] * coeffs
            
        roots_d = np.concatenate(([0.0], np.repeat(z_arr, 2)))
        d_coeffs = np.poly(roots_d)
        
        g_coeffs_rev = g_coeffs[::-1]
        
        q_coeffs, _ = np.polydiv(g_coeffs_rev, d_coeffs)
        
        if len(q_coeffs) == 0 or (len(q_coeffs) == 1 and np.allclose(q_coeffs, 0)):
             return 0.0
             
        q_roots = np.roots(q_coeffs)
        
        real_roots = []
        for r in q_roots:
            if abs(r.imag) < 1e-9:
                real_roots.append(r.real)
                
        if not real_roots:
            return 0.0
            
        max_r = 0.0
        for r in real_roots:
            q_minus = np.polyval(q_coeffs, r - 1e-5)
            q_plus = np.polyval(q_coeffs, r + 1e-5)
            
            if q_minus * q_plus < 0:
                if r > max_r:
                    max_r = r
                    
        B = max(0, max_r)
        return B / (2 * math.pi)
        
    except Exception:
        return float('inf')

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix, binomial
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / math.factorial(k)
        return result
    
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(L_gen_explicit(d, alpha, x))
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0
    
    for k in range(26):
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[1, 0] = 1
    
    for i in range(12):
        zi = z_rationals[i]
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].subs(x, zi)
        b[row_idx, 0] = 0
        
        row_idx = 3 + 2*i
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
        q_poly, _ = g_poly.div(denom_poly)
    except Exception:
        return None
    
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

def _time_remaining(deadline):
    return deadline - time.time()

def construct():
    """Return 12 double-root locations z_1..z_12 using hybrid search."""
    deadline = time.time() + 580.0
    
    # Initial guess: Bessel zeros squared
    z_init = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # Ensure validity
    for i in range(12):
        if not math.isfinite(z_init[i]) or z_init[i] <= 0:
            z_init = [4.0 * (i + 1) for i in range(12)]
            break
    
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_init[i] - z_init[j]) < 1e-12:
                z_init = [4.0 * (i + 1) for i in range(12)]
                break
    
    best_z = list(z_init)
    best_score = _compute_score_fast(best_z)
    
    # Local search using fast evaluation
    np.random.seed(42)
    current_z = best_z
    current_score = best_score
    
    # Try global scales
    scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    for scale in scales:
        if _time_remaining(deadline) < 30:
            break
        candidate = [z * scale for z in current_z]
        score = _compute_score_fast(candidate)
        if score < current_score:
            current_z = candidate
            current_score = score
            best_z = candidate
            best_score = score
    
    # Local perturbations
    step_size = 0.1
    for iteration in range(200):
        if _time_remaining(deadline) < 60:
            break
        
        improved = False
        for i in range(12):
            if _time_remaining(deadline) < 30:
                break
            
            for delta in [step_size, -step_size]:
                candidate = list(current_z)
                candidate[i] *= (1 + delta)
                
                if candidate[i] <= 0:
                    continue
                
                valid = True
                for a in range(12):
                    for b_idx in range(a+1, 12):
                        if abs(candidate[a] - candidate[b_idx]) < 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                score = _compute_score_fast(candidate)
                if score < current_score - 1e-12:
                    current_z = candidate
                    current_score = score
                    best_z = candidate
                    best_score = score
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-4:
                break
    
    # Verify best candidate with exact arithmetic if time permits
    if _time_remaining(deadline) > 120:
        try:
            exact_score = _compute_score_exact(best_z)
            if exact_score is not None:
                best_score = exact_score
        except Exception:
            pass
    
    return best_z
# EVOLVE-BLOCK-END
