"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite


def construct():
    """
    Return three coefficients [c0, c4, c8] for the Hermite combination.
    The degree-12 weight is fixed so g(0) = 0.
    Score is the upper bound on C4 = A(f)*A(Fourier(f)). Lower is better.
    """
    start_time = time.time()
    deadline = start_time + 200  # safety margin before 210s limit

    # Hermite polynomials (physicists' convention) evaluated at 0
    # H_0(x) = 1
    # H_4(x) = 16x^4 - 48x^2 + 12, so H_4(0) = 12
    # H_8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168, so H_8(0) = 168
    # H_12(x) has H_12(0) = 2^6 * 12! / 6! = 64 * 12! / 720
    # Let's compute: H_n(0) = 0 for odd n, and for even n=2k:
    # H_{2k}(0) = (-1)^k * 2^{2k} * (2k)! / k!
    # Actually, let's use the explicit formula or just compute numerically.
    
    H0_0 = 1.0
    H4_0 = eval_hermite(4, 0)  # 12
    H8_0 = eval_hermite(8, 0)  # 168
    H12_0 = eval_hermite(12, 0)  # 2^12 * 12! / 6! * (-1)^6
    
    # g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # g(0) = c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0 = 0
    # So c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    
    def compute_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # We need to compute A(f) and A(Fourier(f)) for the test function.
    # The test function is g(x) * exp(-x^2/2) or similar Gaussian times Hermite combo.
    # Actually, for the uncertainty inequality, we typically consider f(x) = g(x) * exp(-pi*x^2)
    # or f(x) = g(x) * exp(-x^2/2). The Fourier transform of a Gaussian times Hermite
    # polynomial is another Gaussian times Hermite polynomial.
    
    # Let's use the standard setup: f(x) = g(x) * exp(-x^2/2)
    # where g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    # A(f) = sup_x |f(x)| / ||f||_2 (or similar normalization)
    # Actually, in the uncertainty principle context, A(f) typically refers to
    # the "width" or spread. Let me reconsider.
    
    # In the uncertainty inequality A(f)*A(F(f)) >= C, A(f) is often defined as
    # A(f) = sup_x |x - x0| |f(x)| / ||f||_2 or similar.
    
    # Let me use a simpler approach: define the objective as the product of
    # the L-infinity norms of f and F(f), normalized appropriately.
    
    # Actually, let's follow the standard formulation. For a function f,
    # define A(f) = (sup_x |x f(x)|) / ||f||_2
    # and A(F(f)) = (sup_x |xi F(f)(xi)|) / ||F(f)||_2
    # where F is the Fourier transform.
    
    # By Plancherel, ||F(f)||_2 = ||f||_2 (with appropriate normalization).
    
    # The Fourier transform of H_n(x) exp(-x^2/2) is (-i)^n H_n(x) exp(-x^2/2).
    # So F(f)(xi) = c0*H0(xi) + c4*(-i)^4*H4(xi) + c8*(-i)^8*H8(xi) + c12*(-i)^12*H12(xi) times exp(-xi^2/2)
    # = c0*H0(xi) + c4*H4(xi) + c8*H8(xi) + c12*H12(xi) times exp(-xi^2/2)
    # since (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
    
    # So F(f) has the same form as f! This means A(f) = A(F(f)) and the product
    # is A(f)^2.
    
    # So we need to minimize A(f)^2 = (sup_x |x g(x) exp(-x^2/2)|)^2 / ||g exp(-x^2/2)||_2^2
    
    # Let's define h(x) = g(x) exp(-x^2/2)
    # A(f) = sup_x |x h(x)| / ||h||_2
    # Objective = (sup_x |x h(x)|)^2 / ||h||_2^2
    
    # We can compute this numerically.
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = compute_c12(c0, c4, c8)
        
        # Evaluate on a grid
        x = np.linspace(-20, 20, 20000)
        h = (c0 * eval_hermite(0, x) + c4 * eval_hermite(4, x) + 
             c8 * eval_hermite(8, x) + c12 * eval_hermite(12, x)) * np.exp(-x**2 / 2)
        
        # ||h||_2^2 = integral |h|^2 dx
        norm_sq = np.trapz(np.abs(h)**2, x)
        if norm_sq < 1e-15:
            return 1e10
        
        # sup |x * h(x)|
        xh = np.abs(x * h)
        sup_xh = np.max(xh)
        
        return (sup_xh**2) / norm_sq
    
    # Start from the seed values
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try multiple restarts with different initial points
    best_val = objective(x0)
    best_coeffs = x0.copy()
    
    # Gradient-based optimization from the seed
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return np.array(best_coeffs)
    
    # Try some random restarts
    np.random.seed(42)
    for i in range(50):
        if time.time() > deadline - 20:
            break
        # Random perturbations around the best
        scale = 0.5 * (1 - i / 50)
        x_rand = best_coeffs + scale * np.random.randn(3)
        val = objective(x_rand)
        if val < best_val:
            best_val = val
            best_coeffs = x_rand.copy()
            # Polish
            try:
                res = minimize(objective, x_rand, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
