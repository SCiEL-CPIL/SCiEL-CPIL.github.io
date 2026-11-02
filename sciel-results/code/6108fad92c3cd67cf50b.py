import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

def _time_left():
    return _DEADLINE - time.time()

# Precompute Hermite polynomial coefficients [c_0, c_1, ..., c_n] for H_n(x) = sum c_k x^k
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
# H_12 = 4096x^12 - 147456x^10 + 2090112x^8 - 12902400x^6 + 34406400x^4 - 34406400x^2 + 10321920

H0 = np.array([1.0])
H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
H8 = np.array([10080.0, 0.0, -34560.0, 0.0, 23040.0, 0.0, -4608.0, 0.0, 256.0])
H12 = np.array([10321920.0, 0.0, -34406400.0, 0.0, 34406400.0, 0.0, -12902400.0, 0.0, 2090112.0, 0.0, -147456.0, 0.0, 4096.0])

# H_n(0) values
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 3024.0
H12_0 = 10321920.0

def _poly_mul(p, q):
    """Multiply two polynomials represented as coefficient arrays [c0, c1, ...]"""
    if len(p) == 0 or len(q) == 0:
        return np.array([0.0])
    return np.convolve(p, q)

def _poly_deriv(p):
    """Derivative of polynomial [c0, c1, ...] -> [c1, 2c2, ...]"""
    if len(p) <= 1:
        return np.array([0.0])
    return np.arange(1, len(p)) * p[1:]

def _poly_mul_by_x(p):
    """Multiply polynomial by x: [0, c0, c1, ...]"""
    return np.concatenate(([0.0], p))

def _integrate_poly_gaussian(p):
    """
    Compute integral of P(x) * exp(-x^2) dx from -inf to inf.
    P(x) = sum c_k x^k.
    Integral is sum c_k * I_k, where I_k = int x^k exp(-x^2) dx.
    I_k = 0 if k is odd.
    I_{2m} = (2m-1)!! * sqrt(pi) / 2^m.
    """
    result = 0.0
    sqrt_pi = np.sqrt(np.pi)
    for k, c in enumerate(p):
        if c == 0.0:
            continue
        if k % 2 != 0:
            continue
        m = k // 2
        # (2m-1)!! / 2^m * sqrt(pi)
        if m == 0:
            i_k = sqrt_pi
        else:
            # Calculate (2m-1)!!
            double_fact = 1.0
            for i in range(1, 2*m, 2):
                double_fact *= i
            i_k = double_fact * sqrt_pi / (2.0 ** m)
        result += c * i_k
    return result

def _uncertainty_product(a0, a4, a8):
    """
    Compute sigma_x * sigma_p for g(x) = P(x) exp(-x^2/2).
    P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12.
    Constraint: P(0) = 0 => a0 + 12*a4 + 3024*a8 + 10321920*a12 = 0.
    """
    # Calculate a12
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Construct P(x) coefficients
    # Pad arrays to same length for addition
    max_len = max(len(H0), len(H4), len(H8), len(H12))
    P = np.zeros(max_len)
    P[:len(H0)] += a0 * H0
    P[:len(H4)] += a4 * H4
    P[:len(H8)] += a8 * H8
    P[:len(H12)] += a12 * H12
    
    # Check if P is zero
    if np.allclose(P, 0.0):
        return 1e10
        
    # 1. Calculate sigma_x^2
    # Numerator: int x^2 * P(x)^2 * exp(-x^2) dx
    # Denominator: int P(x)^2 * exp(-x^2) dx
    
    P_sq = _poly_mul(P, P)
    x2_P_sq = _poly_mul_by_x(_poly_mul_by_x(P_sq))
    
    num_x = _integrate_poly_gaussian(x2_P_sq)
    den = _integrate_poly_gaussian(P_sq)
    
    if den <= 1e-10:
        return 1e10
        
    sigma_x_sq = num_x / den
    
    # 2. Calculate sigma_p^2
    # g(x) = P(x) exp(-x^2/2)
    # g'(x) = P'(x) exp(-x^2/2) + P(x) (-x) exp(-x^2/2)
    # g'(x) = (P'(x) - x P(x)) exp(-x^2/2)
    # Let Q(x) = P'(x) - x P(x)
    # sigma_p^2 = int |g'|^2 / int |g|^2 = int Q(x)^2 exp(-x^2) / den
    
    P_deriv = _poly_deriv(P)
    x_P = _poly_mul_by_x(P)
    
    # Ensure same length for subtraction
    max_len_q = max(len(P_deriv), len(x_P))
    Q = np.zeros(max_len_q)
    Q[:len(P_deriv)] += P_deriv
    Q[:len(x_P)] -= x_P
    
    Q_sq = _poly_mul(Q, Q)
    num_p = _integrate_poly_gaussian(Q_sq)
    
    sigma_p_sq = num_p / den
    
    if sigma_x_sq < 0 or sigma_p_sq < 0:
        return 1e10
        
    return np.sqrt(sigma_x_sq * sigma_p_sq)

def construct():
    # Seed result
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_result = seed_result.copy()
    best_score = _uncertainty_product(*seed_result)
    
    # Define bounds for optimization
    # The coefficients can vary, but the product is scale-invariant?
    # No, the uncertainty product is scale-invariant.
    # sigma_x * sigma_p is invariant under scaling of g.
    # So we can fix one parameter or normalize.
    # However, the constraint P(0)=0 fixes a12 in terms of a0, a4, a8.
    # The shape is determined by the ratios a0:a4:a8.
    # We can optimize over a0, a4, a8 directly.
    
    bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
    
    # 1. Local refinement from seed
    if _time_left() > 10:
        try:
            res = minimize(
                _uncertainty_product,
                seed_result,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    # 2. Global search with Differential Evolution if time permits
    if _time_left() > 30:
        try:
            de_result = differential_evolution(
                _uncertainty_product,
                bounds,
                seed=42,
                maxiter=500,
                tol=1e-10,
                polish=False,
                init='sobol',
                popsize=20
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x
        except Exception:
            pass
            
    # 3. Final high-precision polish
    if _time_left() > 5:
        try:
            res = minimize(
                _uncertainty_product,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    return best_result
