"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
import time


def construct():
    """
    Returns three coefficients (a0, a4, a8) for the Hermite-Gaussian test function.
    The degree-12 coefficient is fixed to impose g(0) = 0.
    
    Strategy:
    1. Derive the bound formula for A(f)*A(F(f)) in terms of a0, a4, a8.
    2. Use homogeneity to reduce the 3D search to a 1D search over ratios.
    3. Use a high-resolution grid + Brent's method to find the global minimum.
    4. Return the optimal coefficients.
    """
    start_time = time.time()
    deadline = start_time + 180  # 180s safety margin
    
    # Hermite polynomials (probabilists' or physicists'? The task mentions H0, H4, H8, H12)
    # In the context of Fourier-invariant uncertainty principles, we typically use physicists' Hermite polynomials
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 113400x^2 + 10395
    
    # The test function is g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    # With constraint g(0) = 0: a0 + a4*3 + a8*105 + a12*10395 = 0
    # So a12 = -(a0 + 3*a4 + 105*a8) / 10395
    
    # The uncertainty product A(f)*A(F(f)) involves:
    # A(f) = (∫x^2|f(x)|^2 dx / ∫|f(x)|^2 dx)^(1/2)
    # A(F(f)) = (∫ξ^2|F(f)(ξ)|^2 dξ / ∫|F(f)(ξ)|^2 dξ)^(1/2)
    
    # For Hermite-Gaussian functions, these can be computed analytically.
    # The key insight is that the variance product is homogeneous in the coefficients.
    
    # Let's define the moments. For a function g(x) = Σ a_k H_k(x) exp(-x^2/2):
    # ∫|g|^2 dx, ∫x^2|g|^2 dx can be computed using orthogonality of Hermite polynomials.
    
    # Using physicists' Hermite polynomials with weight exp(-x^2):
    # ∫ H_m(x) H_n(x) exp(-x^2) dx = sqrt(pi) 2^n n! δ_mn
    
    # But our weight is exp(-x^2/2), so we need to be careful.
    # Let's use the standard result: for g(x) = Σ a_k H_k(x) exp(-x^2/2),
    # the L2 norm squared is (1/sqrt(2)) Σ a_k^2 2^k k! (using physicists' H_k)
    
    # Actually, let's use a more direct approach. The Fourier transform of H_n(x) exp(-x^2/2)
    # is (-i)^n H_n(ξ) exp(-ξ^2/2) (up to normalization).
    
    # The variance in x-space: <x^2> = ∫ x^2 |g|^2 dx / ∫ |g|^2 dx
    # The variance in frequency space: <ξ^2> = ∫ ξ^2 |ĝ|^2 dξ / ∫ |ĝ|^2 dξ
    
    # For Hermite functions, these can be expressed in terms of the coefficients.
    
    # Let's implement the computation numerically for accuracy, then optimize.
    
    def compute_bound(a0, a4, a8):
        """Compute the upper bound on A(f)*A(F(f)) for given coefficients."""
        if time.time() > deadline:
            return np.inf
        
        # Fix a12 to satisfy g(0) = 0
        a12 = -(a0 + 3*a4 + 105*a8) / 10395
        
        # Define the function and its Fourier transform numerically
        # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
        
        # Use a grid for numerical integration
        x = np.linspace(-15, 15, 2000)
        dx = x[1] - x[0]
        
        # Compute Hermite polynomials
        H0 = np.ones_like(x)
        H4 = x**4 - 6*x**2 + 3
        H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
        H12 = x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 56700*x**4 - 113400*x**2 + 10395
        
        poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
        g = poly * np.exp(-x**2/2)
        
        # Compute L2 norm squared
        norm_sq = np.trapz(np.abs(g)**2, x)
        if norm_sq < 1e-15:
            return np.inf
        
        # Compute <x^2>
        x2_norm = np.trapz(x**2 * np.abs(g)**2, x)
        var_x = x2_norm / norm_sq
        
        # Fourier transform: use FFT for efficiency
        # g(x) = poly(x) * exp(-x^2/2)
        # F(g)(ξ) = (1/sqrt(2π)) ∫ g(x) exp(-iξx) dx
        
        # Use FFT
        N = len(x)
        x_min, x_max = x[0], x[-1]
        dx = x[1] - x[0]
        
        # FFT of g
        G = np.fft.fftshift(np.fft.fft(g)) * dx / np.sqrt(2*np.pi)
        
        # Frequency grid
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        
        # Compute |G|^2
        G_sq = np.abs(G)**2
        
        # L2 norm in frequency space (should equal norm_sq by Parseval)
        norm_freq_sq = np.trapz(G_sq, freqs)
        
        if norm_freq_sq < 1e-15:
            return np.inf
        
        # Compute <ξ^2>
        xi2_norm = np.trapz(freqs**2 * G_sq, freqs)
        var_xi = xi2_norm / norm_freq_sq
        
        # Uncertainty product
        prod = np.sqrt(var_x * var_xi)
        
        return prod
    
    # The bound is homogeneous: if we scale all coefficients by c, the bound doesn't change.
    # So we can fix one coefficient and optimize over the ratios.
    # Let's fix a0 = 1 and optimize over a4/a0 and a8/a0.
    
    # Actually, let's use a 2D search over (a4, a8) with a0 fixed, then scale.
    # But the homogeneity means we can search over the projective space.
    
    # Let's do a grid search over (a4, a8) with a0 = 1, then refine with Brent.
    
    best_bound = np.inf
    best_a0, best_a4, best_a8 = -113/100, 1/25, 1/3240  # Initial guess from seed
    
    # Grid search
    a4_grid = np.linspace(-5, 5, 50)
    a8_grid = np.linspace(-0.5, 0.5, 50)
    
    for a4_val in a4_grid:
        if time.time() > deadline:
            break
        for a8_val in a8_grid:
            if time.time() > deadline:
                break
            bound = compute_bound(1.0, a4_val, a8_val)
            if bound < best_bound:
                best_bound = bound
                best_a0, best_a4, best_a8 = 1.0, a4_val, a8_val
    
    # Refine with Brent's method in 2D (using scipy's minimize_scalar twice)
    if time.time() < deadline:
        # Fix a0 = 1, optimize over a4
        def obj_a4(a4_val):
            return compute_bound(1.0, a4_val, best_a8)
        
        res1 = minimize_scalar(obj_a4, bounds=(best_a4 - 1, best_a4 + 1), method='bounded', 
                              options={'xatol': 1e-8})
        if res1.success:
            best_a4 = res1.x
            best_bound = res1.fun
        
        # Optimize over a8
        def obj_a8(a8_val):
            return compute_bound(1.0, best_a4, a8_val)
        
        res2 = minimize_scalar(obj_a8, bounds=(best_a8 - 0.1, best_a8 + 0.1), method='bounded',
                              options={'xatol': 1e-8})
        if res2.success:
            best_a8 = res2.x
            best_bound = res2.fun
    
    # Return the coefficients (scaled so that a0 is the first value)
    # The task asks for three coefficients: a0, a4, a8
    return np.array([best_a0, best_a4, best_a8])

# EVOLVE-BLOCK-END
