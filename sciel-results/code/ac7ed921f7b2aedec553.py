import numpy as np
from scipy.optimize import minimize
import time
import math

def _get_p_coeffs(roots):
    """
    Returns coefficients of P(x) = prod (x - r_i)^2.
    coeffs[0] is constant term, coeffs[k] is coeff of x^k.
    """
    p_coeffs = [1.0]
    for r in roots:
        # Multiply by (x^2 - 2rx + r^2)
        new_len = len(p_coeffs) + 2
        new_coeffs = [0.0] * new_len
        for i in range(len(p_coeffs)):
            c = p_coeffs[i]
            if c == 0.0:
                continue
            new_coeffs[i + 2] += c
            new_coeffs[i + 1] += -2.0 * r * c
            new_coeffs[i] += r * r * c
        p_coeffs = new_coeffs
    return p_coeffs

def _poly_moment_e_minus_x(coeffs, k):
    """
    Computes \int_0^\infty P(x) x^k e^{-x} dx.
    coeffs: [c0, c1, ...]
    k: integer >= 0
    """
    total = 0.0
    for i in range(len(coeffs)):
        c = coeffs[i]
        if c == 0.0:
            continue
        # \int x^{i+k} e^{-x} dx = (i+k)!
        total += c * math.factorial(i + k)
    return total

def _compute_c4(roots):
    """
    Computes the upper bound on C4.
    Assumption: Bound = \frac{\int P^2 e^{-x}}{(\int P e^{-x})^2}
    """
    p_coeffs = _get_p_coeffs(roots)
    
    # Numerator: \int P(x)^2 e^{-x} dx
    # We need coeffs of P(x)^2.
    # Let's compute P(x)^2 coeffs via convolution.
    n = len(p_coeffs)
    p2_coeffs = [0.0] * (2 * n - 1)
    for i in range(n):
        ci = p_coeffs[i]
        if ci == 0.0:
            continue
        for j in range(n):
            cj = p_coeffs[j]
            if cj == 0.0:
                continue
            p2_coeffs[i + j] += ci * cj
            
    num = _poly_moment_e_minus_x(p2_coeffs, 0)
    
    # Denominator: (\int P(x) e^{-x} dx)^2
    den_inner = _poly_moment_e_minus_x(p_coeffs, 0)
    den = den_inner * den_inner
    
    if den < 1e-300:
        return 1e10
        
    return num / den

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    """
    N = 12
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    # Initial guess
    try:
        x0, w = np.polynomial.laguerre.laggauss(N)
        x0 = np.sort(x0)
    except Exception:
        x0 = np.linspace(1, N*2, N)
        
    def objective(x):
        if np.any(x <= 0) or np.any(np.diff(x) <= 0):
            return 1e10
        try:
            return _compute_c4(x)
        except Exception:
            return 1e10

    def objective_grad(x):
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros(N)
        f0 = objective(x)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = objective(x_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    bounds = [(1e-6, None) for _ in range(N)]
    
    res = minimize(
        objective,
        x0,
        jac=objective_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    if time.time() > deadline - safety_margin:
        return [float(xi) for xi in x0]

    x_opt = res.x
    x_opt = np.sort(x_opt)
    x_opt = np.maximum(x_opt, 1e-6)
    
    if np.any(np.diff(x_opt) <= 0):
        return [float(xi) for xi in x0]
        
    return [float(xi) for xi in x_opt]
