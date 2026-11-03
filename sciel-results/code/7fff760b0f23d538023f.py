import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

# Precompute Hermite polynomials values or coefficients
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 84x^10 + 1260x^8 - 11340x^6 + 30240x^4 - 22680x^2 + 10395

def H0(x):
    return np.ones_like(x)

def H4(x):
    return x**4 - 6*x**2 + 3

def H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def H12(x):
    return x**12 - 84*x**10 + 1260*x**8 - 11340*x**6 + 30240*x**4 - 22680*x**2 + 10395

# We need integrals of the form \int x^4 H_m(x) H_n(x) e^{-x^2} dx
# m, n in {0, 4, 8, 12}
# Due to symmetry, I_{4,m,n} = I_{4,n,m}
# Also H_m are orthogonal with weight e^{-x^2}.
# \int H_m H_n e^{-x^2} = 0 if m != n, \sqrt{\pi} 2^n n! if m=n.
# But we have an extra x^4.
# We can compute these numerically once.

def compute_integral(m_func, n_func):
    # Integrate x^4 * H_m(x) * H_n(x) * exp(-x^2) from -inf to inf
    # Using quad
    def integrand(x):
        return (x**4) * m_func(x) * n_func(x) * np.exp(-x**2)
    
    # The functions are polynomials times Gaussian, so they decay fast.
    # Quad should handle this well.
    val, err = quad(integrand, -np.inf, np.inf, limit=100)
    return val

# Precompute the matrix I
# Indices: 0->H0, 1->H4, 2->H8, 3->H12
H_funcs = [H0, H4, H8, H12]
I = np.zeros((4, 4))
for i in range(4):
    for j in range(i, 4):
        I[i, j] = compute_integral(H_funcs[i], H_funcs[j])
        I[j, i] = I[i, j]

def compute_bound(coeffs):
    c0, c4, c8 = coeffs
    # c12 is determined by g(0) = 0
    # g(0) = (c0*1 + c4*(-3) + c8*105 + c12*(-10395)) * 1 = 0
    # c12 = (c0 - 3c4 + 105c8) / 10395
    c12 = (c0 - 3*c4 + 105*c8) / 10395.0
    
    c = np.array([c0, c4, c8, c12])
    
    # M4(g) = sum_{i,j} c_i c_j I[i,j]
    M4_g = c @ I @ c
    
    # For g_hat, the coefficients are c0, c4, c8, -c12
    # Because Fourier transform of H_n is i^n H_n.
    # i^0=1, i^4=1, i^8=1, i^12=-1.
    c_hat = np.array([c0, c4, c8, -c12])
    
    # M4(g_hat) = sum_{i,j} c_hat_i c_hat_j I[i,j]
    M4_g_hat = c_hat @ I @ c_hat
    
    # The objective is the product
    return M4_g * M4_g_hat

def construct():
    start_time = time.time()
    
    # Initial guess from Goncalves et al.
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Try to optimize
    best_result = x0
    best_score = compute_bound(x0)
    
    # Use L-BFGS-B or Powell
    # Powell is robust for small dimensions
    try:
        res = minimize(compute_bound, x0, method='Powell', options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception as e:
        pass
        
    # Check time
    if time.time() - start_time > 150:
        return best_result
        
    # Try another method for polish
    try:
        res2 = minimize(compute_bound, best_result, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15})
        if res2.success and res2.fun < best_score:
            best_score = res2.fun
            best_result = res2.x
    except Exception as e:
        pass
        
    return best_result
