import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
import time

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 180.0  # Keep under 180s, evaluator kills at 210s

def _get_hermite_coeffs():
    # Probabilists' Hermite polynomials He_n
    # He_0(x) = 1
    # He_2(x) = x^2 - 1
    # He_4(x) = x^4 - 6x^2 + 3
    # He_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # He_12(x) = x^12 - 54x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 55440x^2 + 10395
    
    # We need He_0, He_4, He_8, He_12
    # Note: The task mentions H0, H4, H8, H12. 
    # Standard Hermite polynomials in probability are He_n.
    # In physics, they are H_n. 
    # The seed values suggest specific weights.
    # Let's assume Probabilists' Hermite He_n because the Gaussian is usually e^{-x^2/2} for He_n orthogonality with weight e^{-x^2/2}.
    # If the Gaussian is e^{-x^2}, the weight is different.
    # Goncalves et al. 2017 "Uncertainty in quantum mechanics" often uses e^{-x^2/2}.
    
    # Let's define the polynomials explicitly to avoid library version issues
    def He0(x): return np.ones_like(x)
    def He4(x): return x**4 - 6*x**2 + 3
    def He8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def He12(x): return x**12 - 54*x**10 + 1260*x**8 - 12600*x**6 + 45360*x**4 - 55440*x**2 + 10395
    
    return He0, He4, He8, He12

def _compute_uncertainty_product(c0, c4, c8):
    He0, He4, He8, He12 = _get_hermite_coeffs()
    
    # Calculate c12 to enforce g(0) = 0
    # g(0) = c0*1 + c4*3 + c8*105 + c12*10395 = 0
    # Wait, He4(0) = 3, He8(0) = 105, He12(0) = 10395.
    # c0 + 3*c4 + 105*c8 + 10395*c12 = 0
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    
    # Grid for numerical integration
    # We need to compute variances of |g(x)|^2
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    
    # Gaussian factor
    # Assume g(x) = (poly(x)) * exp(-x^2/2)
    # Then |g(x)|^2 = poly(x)^2 * exp(-x^2)
    poly = c0*He0(x) + c4*He4(x) + c8*He8(x) + c12*He12(x)
    prob_density = poly**2 * np.exp(-x**2)
    
    # Normalize
    norm = np.sum(prob_density) * dx
    if norm < 1e-12:
        return 1e10
    
    prob_density /= norm
    
    # Calculate x^2 expectation
    x2_mean = np.sum(x**2 * prob_density) * dx
    
    # For Fourier transform of g(x) = P(x) e^{-x^2/2}:
    # If g is a sum of Hermite functions, its Fourier transform is also a sum of Hermite functions (with phase factors).
    # He_n(x) e^{-x^2/2} <-> i^n He_n(k) e^{-k^2/2} (up to normalization constants)
    # Specifically, F[He_n(x) e^{-x^2/2}] = i^n He_n(k) e^{-k^2/2}
    # So the probability density in frequency domain |G(k)|^2 is proportional to |poly(i*k)|^2? No.
    # The Fourier transform of He_n(x) e^{-x^2/2} is i^n He_n(k) e^{-k^2/2}.
    # So if g(x) = \sum c_n He_n(x) e^{-x^2/2}, then G(k) = \sum c_n i^n He_n(k) e^{-k^2/2}.
    # The magnitude squared |G(k)|^2 will have the same "shape" distribution as |g(x)|^2 if the coefficients are real?
    # Not exactly. The coefficients in frequency domain are $c_n i^n$.
    # Since $i^n$ is a phase, $|G(k)|^2$ is determined by the coefficients $c_n$ mixed via interference.
    
    # Let's compute G(k) numerically or analytically.
    # G(k) = (c0*He0(k) + c4*i^4*He4(k) + c8*i^8*He8(k) + c12*i^12*He12(k)) * e^{-k^2/2}
    # i^4 = 1, i^8 = 1, i^12 = -1.
    # So G(k) = (c0*He0(k) + c4*He4(k) + c8*He8(k) - c12*He12(k)) * e^{-k^2/2}
    
    k = x # Symmetric grid
    poly_k = c0*He0(k) + c4*He4(k) + c8*He8(k) - c12*He12(k)
    prob_density_k = poly_k**2 * np.exp(-k**2)
    
    norm_k = np.sum(prob_density_k) * dx
    if norm_k < 1e-12:
        return 1e10
        
    prob_density_k /= norm_k
    
    k2_mean = np.sum(k**2 * prob_density_k) * dx
    
    sigma_x = np.sqrt(x2_mean)
    sigma_k = np.sqrt(k2_mean)
    
    return sigma_x * sigma_k

def construct():
    # Initial guess from seed
    c0_init = -113 / 100
    c4_init = 1 / 25
    c8_init = 1 / 3240
    
    x0 = np.array([c0_init, c4_init, c8_init])
    
    def objective(vars):
        return _compute_uncertainty_product(vars[0], vars[1], vars[2])
    
    # Try to optimize
    best_val = objective(x0)
    best_x = x0
    
    # Check time
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 10:
        return best_x
    
    try:
        # Use Nelder-Mead for robustness, then polish with L-BFGS-B
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
        # Polish
        elapsed = time.time() - START_TIME
        if elapsed < TIME_LIMIT - 5:
            res2 = minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
                
    except Exception:
        pass
    
    return np.array(best_x)
