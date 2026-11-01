"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, SLSQP
from scipy.special import factorial
import math

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 180s limit, use 210s wall-clock but be safe
    
    # The task is to find coefficients c0, c4, c8 for a function g(x) = c0*H0(x)*e^{-x^2} + c4*H4(x)*e^{-x^2} + c8*H8(x)*e^{-x^2}
    # such that g(0) = 0 (which determines c12 if we had H12, but here we only have 3 free coeffs)
    # Wait, the task says: "Three free coefficients multiply H0,H4,H8; the notebook derives the H12 coefficient to impose P(0)=0."
    # But the interface says: "construct() -> three coefficients of the Hermite combination (degrees 0, 4, 8; the degree-12 weight is fixed so g(0) = 0)"
    # So we return [c0, c4, c8], and there's an implicit c12 determined by g(0)=0.
    
    # Let's understand the problem better. We have a test function g(x) = sum_{k in {0,4,8,12}} c_k H_k(x) e^{-x^2}
    # with c12 determined by g(0) = 0.
    # H_k(0) = 0 for odd k, and for even k: H_{2n}(0) = (-1)^n (2n)!/n!
    # So g(0) = c0*H0(0)*e^0 + c4*H4(0)*e^0 + c8*H8(0)*e^0 + c12*H12(0)*e^0 = 0
    # H0(0) = 1
    # H4(0) = -3 (let me verify: H4(x) = 16x^4 - 48x^2 + 12, so H4(0) = 12? No wait)
    # Actually, let me use the physicists' Hermite polynomials:
    # H0(x) = 1
    # H2(x) = 4x^2 - 2
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13824x^2 + 3024
    # H12(x) = ...
    # 
    # H4(0) = 12
    # H8(0) = 3024
    # H12(0) = ?
    # 
    # The recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    # H0(0) = 1
    # H1(0) = 0
    # H2(0) = 2*0*H1(0) - 2*1*H0(0) = -2
    # H3(0) = 2*0*H2(0) - 2*2*H1(0) = 0
    # H4(0) = 2*0*H3(0) - 2*3*H2(0) = -6*(-2) = 12
    # H5(0) = 0
    # H6(0) = -2*5*H4(0)/... wait, H_{n+1}(0) = -2n H_{n-1}(0) for n>=1
    # H6(0) = -2*5*H4(0) = -10*12 = -120
    # H8(0) = -2*7*H6(0) = -14*(-120) = 1680
    # H10(0) = -2*9*H8(0) = -18*1680 = -30240
    # H12(0) = -2*11*H10(0) = -22*(-30240) = 665280
    # 
    # So g(0) = c0*1 + c4*12 + c8*1680 + c12*665280 = 0
    # => c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    
    # The score is the upper bound on C4 = A(f)*A(Fourier(f)) where A(g) = ||g||_2 / ||g||_∞ or similar.
    # Actually, in uncertainty principles, A(f) typically denotes the uncertainty measure.
    # For the Fourier transform, the product of uncertainties is bounded below by 1/4 (Heisenberg).
    # But here it's about an upper bound on A(f)*A(Fourier(f)) for a specific class of functions.
    
    # Let me think about what A(f) means. In many contexts, A(f) = sqrt(<x^2> - <x>^2) * sqrt(<p^2> - <p>^2)
    # where <x^2> = ∫ x^2 |f(x)|^2 dx / ∫ |f(x)|^2 dx
    # 
    # Actually, looking at the seed program, it returns [-113/100, 1/25, 1/3240].
    # The score is "the upper bound on C4" and "lower is better".
    # 
    # Let me try to compute the uncertainty product for a given set of coefficients.
    # 
    # g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * e^{-x^2}
    # 
    # The norm ||g||_2^2 = ∫ g(x)^2 dx
    # The moments <x^2> = ∫ x^2 g(x)^2 dx / ||g||_2^2
    # Similarly for the Fourier transform.
    
    # Since the current score is 0.0000, the seed program is failing. Let me try a simple optimization.
    
    # Let me define the Hermite polynomials and compute the uncertainty product numerically.
    
    def hermite(n, x):
        """Compute the nth physicist's Hermite polynomial at x."""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            H_nm1 = hermite(n-1, x)
            H_nm2 = hermite(n-2, x)
            return 2 * x * H_nm1 - 2 * (n-1) * H_nm2
    
    def compute_score(c0, c4, c8):
        """Compute the uncertainty product for given coefficients."""
        c12 = -(c0 + 12*c4 + 1680*c8) / 665280.0
        
        # Use a grid for numerical integration
        x = np.linspace(-15, 15, 10000)
        dx = x[1] - x[0]
        
        H0 = hermite(0, x)
        H4 = hermite(4, x)
        H8 = hermite(8, x)
        H12 = hermite(12, x)
        
        g = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x**2)
        
        g2 = g**2
        norm2 = np.sum(g2) * dx
        if norm2 <= 0:
            return 1e10
        
        # Position variance
        x2_g2 = x**2 * g2
        x2_moment = np.sum(x2_g2) * dx / norm2
        x_moment = np.sum(x * g) * dx / norm2  # This is <x> for |g|^2, but we need <x> from g*g
        # Actually, <x> = ∫ x |g(x)|^2 dx / ∫ |g(x)|^2 dx
        x_moment = np.sum(x * g2) * dx / norm2
        var_x = x2_moment - x_moment**2
        
        # For the Fourier transform, we can use the fact that the Fourier transform of H_n(x) e^{-x^2}
        # is related to H_n(-k) e^{-k^2} (up to a phase and normalization).
        # Actually, the Fourier transform of H_n(x) e^{-x^2/2} is i^n H_n(k) e^{-k^2/2} (with appropriate convention).
        # Let me use the convention: F(k) = ∫ f(x) e^{-2πixk} dx
        # 
        # Alternatively, we can compute the Fourier transform numerically.
        # 
        # Let's use the property that for f(x) = H_n(x) e^{-x^2}, 
        # the Fourier transform is proportional to H_n(k) e^{-k^2/4} (with some scaling).
        # 
        # Actually, let me just compute it numerically using FFT or direct integration.
        # 
        # For efficiency, let me use the fact that the momentum variance can be computed from the derivatives.
        # <p^2> = ∫ |f'(x)|^2 dx / ∫ |f(x)|^2 dx (up to a constant depending on the Fourier convention).
        # 
        # With the convention F(k) = ∫ f(x) e^{-ikx} dx, we have <p^2> = ∫ |f'(x)|^2 dx / ||f||_2^2.
        
        # Compute g'(x)
        # g(x) = P(x) e^{-x^2} where P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        # g'(x) = P'(x) e^{-x^2} - 2x P(x) e^{-x^2} = (P'(x) - 2x P(x)) e^{-x^2}
        
        # P'(x) = c0*H0'(x) + c4*H4'(x) + c8*H8'(x) + c12*H12'(x)
        # H_n'(x) = 2n H_{n-1}(x)
        
        H0p = np.zeros_like(x)
        H4p = 8 * hermite(3, x)
        H8p = 16 * hermite(7, x)
        H12p = 24 * hermite(11, x)
        
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        Pp = c0*H0p + c4*H4p + c8*H8p + c12*H12p
        
        gp = (Pp - 2*x*P) * np.exp(-x**2)
        
        gp2 = gp**2
        norm_p2 = np.sum(gp2) * dx
        
        if norm_p2 <= 0:
            return 1e10
        
        var_p = norm_p2 / norm2  # This is <p^2> since <p> should be 0 by symmetry
        
        # The uncertainty product is sqrt(var_x * var_p)
        # But we need to be careful about the Fourier convention.
        # With F(k) = ∫ f(x) e^{-ikx} dx, the uncertainty product is sqrt(var_x * var_p).
        
        uncertainty_product = math.sqrt(var_x * var_p)
        
        return uncertainty_product
    
    # Try the seed values first
    seed_c0, seed_c4, seed_c8 = -113/100, 1/25, 1/3240
    seed_score = compute_score(seed_c0, seed_c4, seed_c8)
    best_score = seed_score
    best_c = np.array([seed_c0, seed_c4, seed_c8])
    
    # Try some simple optimizations
    # Use scipy.optimize.minimize
    def objective(c):
        if time.time() > deadline - 30:
            return best_score
        return compute_score(c[0], c[1], c[2])
    
    # Try a few random starting points
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        
        x0 = np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5)
        ])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Try around the seed
    x0_seed = np.array([seed_c0, seed_c4, seed_c8])
    try:
        result = minimize(objective, x0_seed, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_score:
            best_score = result.fun
            best_c = result.x
    except Exception:
        pass
    
    return best_c

# EVOLVE-BLOCK-END
