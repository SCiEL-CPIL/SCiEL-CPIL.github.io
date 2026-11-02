"""Rewritten program for the Uncertainty Inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def _hermite_poly(n, x):
    """Compute the physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        return 256 * x**8 - 4608 * x**6 + 23040 * x**4 - 34560 * x**2 + 13440
    elif n == 12:
        return 4096 * x**12 - 147456 * x**10 + 2580480 * x**8 - 18903040 * x**6 + 68520960 * x**4 - 108973440 * x**2 + 47900160
    else:
        raise ValueError(f"Hermite polynomial of degree {n} not implemented")

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound score for the given coefficients.
    The H12 coefficient is derived to satisfy g(0) = 0.
    g(x) = exp(-pi x^2) * (c0*H0 + c4*H4 + c8*H8 + c12*H12)
    g(0) = c0*1 + c4*12 + c8*13440 + c12*47900160 = 0
    """
    # Calculate c12
    c12 = -(c0 * 1.0 + c4 * 12.0 + c8 * 13440.0) / 47900160.0
    
    # Define the polynomial part P(x)
    def P(x):
        return c0 * _hermite_poly(0, x) + c4 * _hermite_poly(4, x) + c8 * _hermite_poly(8, x) + c12 * _hermite_poly(12, x)
    
    # Define g(x)
    def g(x):
        return np.exp(-np.pi * x**2) * P(x)
    
    # Calculate L4 norm of g
    # Integral of |g(x)|^4 from -inf to inf
    def integrand_g4(x):
        val = g(x)
        return val**4
    
    # Use quad for numerical integration
    # Symmetry: g(x) is even if all coefficients are for even Hermite polynomials (which they are).
    # H0, H4, H8, H12 are even. So g(x) is even.
    # We can integrate from 0 to inf and multiply by 2.
    try:
        norm_g4_sq, _ = quad(integrand_g4, 0, np.inf, limit=100)
        norm_g4_sq *= 2
    except Exception:
        return 1e10 # Return a large value if integration fails
    
    # Calculate Fourier transform \hat{g}(\xi)
    # Since g(x) = e^{-pi x^2} P(x), and P(x) is a polynomial, \hat{g}(\xi) is also a polynomial times e^{-pi \xi^2}.
    # We can compute \hat{g} numerically or analytically.
    # Analytical derivation of \hat{g} is complex. Let's use the property that Fourier transform of e^{-pi x^2} H_n(x) is (-i)^n e^{-pi \xi^2} H_n(\xi) (with appropriate normalization).
    # Actually, for physicists' Hermite polynomials:
    # \mathcal{F}[e^{-\pi x^2} H_n(x)] = (-i)^n e^{-\pi \xi^2} H_n(\xi)
    # Let's verify for n=0: \mathcal{F}[e^{-\pi x^2}] = e^{-\pi \xi^2}. Correct.
    # n=1: \mathcal{F}[2x e^{-\pi x^2}] = -2i \mathcal{F}[x e^{-\pi x^2}]? 
    # \frac{d}{d\xi} e^{-\pi \xi^2} = -2\pi \xi e^{-\pi \xi^2}.
    # \mathcal{F}[x f(x)] = \frac{i}{2\pi} \frac{d}{d\xi} \hat{f}(\xi).
    # \mathcal{F}[x e^{-\pi x^2}] = \frac{i}{2\pi} (-2\pi \xi e^{-\pi \xi^2}) = -i \xi e^{-\pi \xi^2}.
    # So \mathcal{F}[2x e^{-\pi x^2}] = -2i \xi e^{-\pi \xi^2} = (-i)^1 e^{-\pi \xi^2} H_1(\xi). Correct.
    
    # So \hat{g}(\xi) = e^{-\pi \xi^2} [ c0 (-i)^0 H0(\xi) + c4 (-i)^4 H4(\xi) + c8 (-i)^8 H8(\xi) + c12 (-i)^12 H12(\xi) ]
    # (-i)^0 = 1
    # (-i)^4 = 1
    # (-i)^8 = 1
    # (-i)^12 = 1
    # So \hat{g}(\xi) = e^{-\pi \xi^2} [ c0 H0(\xi) + c4 H4(\xi) + c8 H8(\xi) + c12 H12(\xi) ]
    # It turns out the Fourier transform is the same function! (Because all degrees are multiples of 4).
    
    def g_hat(xi):
        return np.exp(-np.pi * xi**2) * (c0 * _hermite_poly(0, xi) + c4 * _hermite_poly(4, xi) + c8 * _hermite_poly(8, xi) + c12 * _hermite_poly(12, xi))
    
    def integrand_g_hat4(xi):
        val = g_hat(xi)
        return val**4

    try:
        norm_g_hat4_sq, _ = quad(integrand_g_hat4, 0, np.inf, limit=100)
        norm_g_hat4_sq *= 2
    except Exception:
        return 1e10
    
    if norm_g4_sq <= 0 or norm_g_hat4_sq <= 0:
        return 1e10
        
    # The score is likely the product of the L4 norms or a ratio.
    # "Upper bound on A(f)A(Fourier(f))"
    # If A(f) = ||f||_4, then A(f)A(hat{f}) = ||f||_4 ||hat{f}||_4.
    # We want to minimize this product? Or is it a bound on the ratio?
    # Given "lower is better", minimizing the product makes sense if it's a measure of "spread" or "uncertainty" in a multiplicative sense.
    # However, usually uncertainty is lower bounded.
    # Let's assume the score is the product ||g||_4 * ||hat{g}||_4.
    score = np.sqrt(norm_g4_sq) * np.sqrt(norm_g_hat4_sq)
    
    return score

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 170.0 # Safety margin
    
    # Initial guess (seed)
    c0_init, c4_init, c8_init = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    x0 = np.array([c0_init, c4_init, c8_init])
    
    def objective(x):
        c0, c4, c8 = x
        return _compute_bound(c0, c4, c8)
    
    best_score = objective(x0)
    best_x = x0
    
    # Try a few different optimization methods and starting points
    # Method 1: L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500})
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() - start_time > deadline:
        return np.array(best_x)
        
    # Method 2: Nelder-Mead (more robust to non-smoothness, though function should be smooth)
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass
        
    if time.time() - start_time > deadline:
        return np.array(best_x)

    # Try a perturbation search if optimization didn't improve much
    # Random restarts
    np.random.seed(42)
    for _ in range(10):
        if time.time() - start_time > deadline - 5.0:
            break
        x_rand = x0 + np.random.normal(0, 0.1, 3)
        try:
            res = minimize(objective, x_rand, method='L-BFGS-B', options={'maxiter': 200})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            continue

    return np.array(best_x)

# EVOLVE-BLOCK-END
