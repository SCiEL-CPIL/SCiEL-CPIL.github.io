import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite, factorial

START_TIME = time.time()
TIME_LIMIT = 205.0  # Safety margin before 210s

# Precompute Hermite polynomial values and derivatives at key points
# H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# We need H_0, H_4, H_8, H_12

# Values at x=0:
# H_0(0) = 1
# H_4(0) = 3
# H_8(0) = 105
# H_12(0) = 945
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 945.0

# For the uncertainty bound, we need to compute:
# A(g) = sup_x |x * g(x)|  and  A(hat{g}) = sup_x |xi * hat{g}(xi)|
# The product A(g)*A(hat{g}) is bounded by C4 * (some expression)
# 
# For a Hermite-Gaussian function g(x) = P(x) * exp(-x^2/2),
# where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12,
# the Fourier transform is also a Hermite-Gaussian.
#
# The key insight is that for such functions, the uncertainty product
# can be computed analytically using the moments of |g|^2 and |hat{g}|^2.

def _hermite_coeffs():
    """Return coefficients of H0, H4, H8, H12 in power basis."""
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 226800x^2 + 945
    # 
    # In the basis [x^0, x^2, x^4, x^6, x^8, x^10, x^12]:
    h0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    h4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    h8 = np.array([105.0, -420.0, 210.0, 0.0, 1.0, 0.0, 0.0])
    h12 = np.array([945.0, -226800.0, 151200.0, -30240.0, 2520.0, -84.0, 1.0])
    return h0, h4, h8, h12

# Precompute integral values: integral of x^{2n} * exp(-x^2) dx from -inf to inf
# = sqrt(pi) * (2n)! / (4^n * n!)
_INTEGRAL_CACHE = {}
def _integral_x2n(n):
    """Integral of x^{2n} * exp(-x^2) dx from -inf to inf, cached."""
    if n in _INTEGRAL_CACHE:
        return _INTEGRAL_CACHE[n]
    if n == 0:
        val = np.sqrt(np.pi)
    else:
        # (2n)! / (4^n * n!) * sqrt(pi)
        log_val = np.sum(np.log(np.arange(1, 2*n+1))) - n*np.log(4) - np.sum(np.log(np.arange(1, n+1))) + 0.5*np.log(np.pi)
        val = np.exp(log_val)
    _INTEGRAL_CACHE[n] = val
    return val

# Precompute all needed integrals up to n=13 (for R^2 which has degree 26, so n up to 13)
for _n in range(14):
    _integral_x2n(_n)

def _compute_moments(c0, c4, c8):
    """
    Compute the moments needed for the uncertainty bound.
    
    For g(x) = P(x) * exp(-x^2/2), we need:
    - ||g||^2 = integral of |g(x)|^2 dx
    - ||x*g||^2 = integral of x^2 * |g(x)|^2 dx
    - ||hat{g}||^2 = ||g||^2 (Plancherel)
    - ||xi*hat{g}||^2 = ||g'||^2 (Fourier derivative property)
    
    The uncertainty product is bounded by:
    A(g)*A(hat{g}) <= C4 * (||x*g||^2 / ||g||^2)^(1/2) * (||g'||^2 / ||g||^2)^(1/2)
    """
    h0, h4, h8, h12 = _hermite_coeffs()
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # c12 = -(c0 + 3*c4 + 105*c8) / 945
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    
    # P(x) in power basis [x^0, x^2, x^4, x^6, x^8, x^10, x^12]
    P = c0*h0 + c4*h4 + c8*h8 + c12*h12
    
    # P(x)^2 is a polynomial of degree 24
    # Let Q[m] be the coefficient of x^{2m} in P(x)^2
    Q = np.zeros(13)  # degrees 0 to 24
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    
    # ||g||^2 = sum_{m=0}^{12} Q[m] * integral_x2n(m)
    norm_sq = 0.0
    for m in range(13):
        norm_sq += Q[m] * _INTEGRAL_CACHE[m]
    
    # ||x*g||^2 = integral x^2 * P(x)^2 * exp(-x^2) dx
    # = sum_{m=0}^{12} Q[m] * integral x^{2(m+1)} exp(-x^2) dx
    x_norm_sq = 0.0
    for m in range(13):
        x_norm_sq += Q[m] * _INTEGRAL_CACHE[m+1]
    
    # Now compute ||g'||^2
    # g(x) = P(x) * exp(-x^2/2)
    # g'(x) = P'(x) * exp(-x^2/2) - x * P(x) * exp(-x^2/2)
    # g'(x) = [P'(x) - x*P(x)] * exp(-x^2/2)
    
    # P'(x) = sum_{k=1}^{6} 2k * P[2k] * x^{2k-1}
    # x*P(x) = sum_{k=0}^{6} P[2k] * x^{2k+1}
    
    # Let R(x) = P'(x) - x*P(x)
    # R(x) is an odd polynomial
    
    # R(x) = sum_{k=0}^{6} R[2k+1] * x^{2k+1}
    R = np.zeros(7)  # coefficients for x^1, x^3, ..., x^13
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    
    # |g'(x)|^2 = R(x)^2 * exp(-x^2)
    # R(x)^2 is an even polynomial of degree 26
    # Let S[m] be the coefficient of x^{2m} in R(x)^2
    S = np.zeros(14)  # degrees 0 to 26
    for i in range(7):
        for j in range(7):
            S[i+j] += R[i] * R[j]
    
    # ||g'||^2 = sum_{m=0}^{13} S[m] * integral_x2n(m)
    deriv_norm_sq = 0.0
    for m in range(14):
        deriv_norm_sq += S[m] * _INTEGRAL_CACHE[m]
    
    # The uncertainty product bound
    # A(g) * A(hat{g}) <= C4 * sqrt(||x*g||^2 / ||g||^2) * sqrt(||g'||^2 / ||g||^2)
    # = C4 * sqrt(||x*g||^2 * ||g'||^2 / ||g||^2^2)
    
    if norm_sq <= 0:
        return 0.0, 0.0, 0.0  # Degenerate case
    
    ratio_x = x_norm_sq / norm_sq
    ratio_d = deriv_norm_sq / norm_sq
    
    bound = np.sqrt(ratio_x * ratio_d)
    
    return bound, norm_sq, x_norm_sq

def _objective(coeffs):
    """Objective function to minimize."""
    bound, _, _ = _compute_moments(coeffs[0], coeffs[1], coeffs[2])
    return bound

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Seed initial guess
    seed = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = _objective(seed)
    best_coeffs = seed.copy()
    
    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.5, 0.5)]
    
    # Phase 1: Differential evolution for global exploration
    if time.time() < TIME_LIMIT - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-10,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=True
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B with finite differences
    num_starts = 20
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 30:
            break
        
        # Generate diverse initial guesses
        if i == 0:
            x0 = best_coeffs.copy()
        elif i == 1:
            x0 = seed.copy()
        elif i == 2:
            # Try symmetric points
            x0 = np.array([0.0, 0.0, 0.0])
        elif i == 3:
            x0 = np.array([-1.0, 0.1, 0.01])
        elif i == 4:
            x0 = np.array([1.0, -0.1, -0.01])
        else:
            x0 = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1])
            ])
        
        try:
            res = minimize(_objective, x0, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Targeted refinement near best solution
    for i in range(15):
        if time.time() > TIME_LIMIT - 15:
            break
        
        # Perturb the best solution with decreasing scale
        scale = 0.1 * (0.5 ** (i // 5))
        x0 = best_coeffs + np.random.normal(0, scale, size=3)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(_objective, x0, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < TIME_LIMIT - 10:
        try:
            res = minimize(_objective, best_coeffs, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
