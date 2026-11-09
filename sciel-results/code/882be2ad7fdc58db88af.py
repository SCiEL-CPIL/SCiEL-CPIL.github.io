"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from sympy import Rational, Matrix, Poly, real_roots, pi as sympy_pi, binomial, symbols, factor

# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from sympy import Rational, Matrix, Poly, real_roots, pi as sympy_pi, binomial, symbols, factor
from scipy.special import roots_genlaguerre

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728  # j_{0,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _L_val(n, alpha, z):
    """Compute L_n^(alpha)(z) exactly for rational z and alpha."""
    result = Rational(0)
    for k in range(n + 1):
        fk = math.factorial(k)
        term = binomial(n + alpha, n - k) * ((-z) ** k) / fk
        result += term
    return result

def _L_deriv_val(n, alpha, z):
    """Compute L_n^(alpha)'(z) = -L_{n-1}^(alpha+1)(z)."""
    if n == 0:
        return Rational(0)
    return -_L_val(n - 1, alpha + 1, z)

def _get_laguerre_coeffs(n, alpha):
    """Return list of coefficients [c_0, c_1, ..., c_n] for L_n^(alpha)(x) = sum c_i x^i."""
    coeffs = [Rational(0)] * (n + 1)
    for k in range(n + 1):
        fk = math.factorial(k)
        coeffs[k] = binomial(n + alpha, n - k) * ((-1) ** k) / fk
    return coeffs

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Pre-compute Laguerre polynomial coefficients in monomial basis
    lag_coeffs = []
    for k in range(26):
        n = 2 * k
        lag_coeffs.append(_get_laguerre_coeffs(n, alpha))
    
    # Build matrix A and vector b
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    points = [Rational(0)] + z_rationals
    
    vals = [[0]*len(points) for _ in range(26)]
    derivs = [[0]*len(points) for _ in range(26)]
    
    for k in range(26):
        n = 2 * k
        for i, p in enumerate(points):
            vals[k][i] = _L_val(n, alpha, p)
            derivs[k][i] = _L_deriv_val(n, alpha, p)
            
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = vals[k][0]
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = derivs[k][0]
    b[1, 0] = 1
    
    # Rows 2..25
    for i in range(12):
        point_idx = i + 1
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = vals[k][point_idx]
        b[row_idx, 0] = 0
        
        row_idx = 3 + 2*i
        for k in range(26):
            A[row_idx, k] = derivs[k][point_idx]
        b[row_idx, 0] = 0
        
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    max_deg = 50
    G = [Rational(0)] * (max_deg + 1)
    for k in range(26):
        ck = c[k]
        coeffs_k = lag_coeffs[k]
        for j in range(len(coeffs_k)):
            G[j] += ck * coeffs_k[j]
            
    D = [Rational(0)] * (2 * 12 + 1)
    D[1] = 1
    for zi in z_rationals:
        new_D = [Rational(0)] * (len(D) + 2)
        for i in range(len(D)):
            new_D[i] += D[i] * (zi**2)
            new_D[i+1] += D[i] * (-2 * zi)
            new_D[i+2] += D[i] * 1
        D = new_D
        
    q_coeffs = [Rational(0)] * (max_deg - len(D) + 1)
    rem = G[:]
    
    for i in range(len(q_coeffs)):
        lead_idx = None
        for j in range(len(rem)):
            if rem[j] != 0:
                lead_idx = j
                break
        if lead_idx is None:
            break
            
        if D[-1] == 0:
            return None
            
        coeff = rem[lead_idx] / D[-1]
        q_coeffs[i] = coeff
        
        for j in range(len(D)):
            rem[lead_idx + j] -= coeff * D[j]
            
    # Check remainder
    for val in rem:
        if val != 0:
            pass
            
    q_poly = Poly(sum(q_coeffs[i] * x**i for i in range(len(q_coeffs))), x)
    try:
        real_roots_list = real_roots(q_poly)
    except Exception:
        return None
        
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        try:
            q_at_minus = q_poly.eval(r - eps)
            q_at_plus = q_poly.eval(r + eps)
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
    """Return 12 double-root locations z_1..z_12 using Laguerre nodes."""
    deadline = time.time() + 570.0
    
    # Use roots of generalized Laguerre polynomial L_12^(-1/2)(x)
    try:
        nodes, weights = roots_genlaguerre(12, -0.5)
        result = [float(n) for n in nodes]
    except Exception:
        # Fallback to Bessel zeros squared if scipy fails
        result = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
        
    # Ensure positive, finite, distinct
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            result = [4.0 * (i + 1) for i in range(12)]
            break
            
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                result = [4.0 * (i + 1) for i in range(12)]
                break
                
    best_score = float('inf')
    best_result = result
    
    # Evaluate initial guess
    remaining = _time_remaining(deadline)
    if remaining > 120:
        score = _compute_score_exact(result)
        if score is not None:
            best_score = score
            best_result = list(result)
            
    # Try scaling factors if time permits
    scale_factors = [0.8, 0.9, 1.1, 1.2, 1.5, 0.7, 1.3, 2.0, 0.5]
    
    for s in scale_factors:
        if _time_remaining(deadline) < 90:
            break
        candidate = [z * s for z in best_result]
        valid = True
        for i in range(12):
            if not math.isfinite(candidate[i]) or candidate[i] <= 0:
                valid = False
                break
        if valid:
            for i in range(12):
                for j in range(i+1, 12):
                    if abs(candidate[i] - candidate[j]) < 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        if not valid:
            continue
            
        score = _compute_score_exact(candidate)
        if score is not None and score < best_score:
            best_score = score
            best_result = candidate
            
    return best_result
# EVOLVE-BLOCK-END
