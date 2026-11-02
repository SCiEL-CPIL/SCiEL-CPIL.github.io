import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import math

def H0(x):
    return 1

def H4(x):
    return x**4 - 6*x**2 + 3

def H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def H12(x):
    return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 45360*x**4 - 55440*x**2 + 10395

def a12_calc(a0, a4, a8):
    return -(a0 + 3*a4 + 105*a8) / 10395.0

def g_func(x, a0, a4, a8):
    a12 = a12_calc(a0, a4, a8)
    return (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * np.exp(-x**2 / 2.0)

def g_hat_func(x, a0, a4, a8):
    a12 = a12_calc(a0, a4, a8)
    # Fourier transform coefficients: a0, a4, a8, -a12
    return (a0*H0(x) + a4*H4(x) + a8*H8(x) - a12*H12(x)) * np.exp(-x**2 / 2.0)

def objective(params):
    a4, a8 = params
    a0 = 1.0
    
    # Calculate I2 for g using orthogonality
    a12 = a12_calc(a0, a4, a8)
    norm_sq = (a0**2 * math.sqrt(math.pi) * 2**0 * math.factorial(0) +
               a4**2 * math.sqrt(math.pi) * 2**4 * math.factorial(4) +
               a8**2 * math.sqrt(math.pi) * 2**8 * math.factorial(8) +
               a12**2 * math.sqrt(math.pi) * 2**12 * math.factorial(12))
    
    # Calculate I4 for g
    def integrand_g(x):
        val = g_func(x, a0, a4, a8)
        return val**4
    
    I4_g, _ = quad(integrand_g, -20, 20, limit=200, epsabs=1e-14, epsrel=1e-14)
    
    # Calculate I4 for g_hat
    def integrand_ghat(x):
        val = g_hat_func(x, a0, a4, a8)
        return val**4
    
    I4_ghat, _ = quad(integrand_ghat, -20, 20, limit=200, epsabs=1e-14, epsrel=1e-14)
    
    # Minimize (I4_g * I4_ghat) / norm_sq**4 which is equivalent to minimizing the 4th power of the ratio
    val = (I4_g * I4_ghat) / (norm_sq**4)
    return val

def construct():
    # Initial guess from seed
    seed_a0 = -113.0 / 100.0
    seed_a4 = 1.0 / 25.0
    seed_a8 = 1.0 / 3240.0
    
    init_a4 = seed_a4 / seed_a0
    init_a8 = seed_a8 / seed_a0
    
    # Try multiple initial points and use the best one
    initial_points = [
        [init_a4, init_a8],
        [0.0, 0.0],
        [0.1, -0.01],
        [-0.1, 0.01],
        [0.05, 0.005],
        [-0.05, -0.005],
    ]
    
    best_result = None
    best_value = float('inf')
    
    for init in initial_points:
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 5000, 'maxfev': 5000})
            if res.fun < best_value:
                best_value = res.fun
                best_result = res
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to seed values
        return np.array([1.0, init_a4, init_a8])
    
    best_a4, best_a8 = best_result.x
    best_a0 = 1.0
    
    return np.array([best_a0, best_a4, best_a8])
