"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial

def construct():
    """
    Returns three coefficients (c0, c4, c8) for the Hermite combination.
    The degree-12 weight is fixed so that g(0) = 0.
    We minimize the uncertainty product upper bound.
    """
    start_time = time.time()
    deadline = start_time + 200  # 10s safety margin before 210s kill
    
    # Hermite polynomials (physicists' convention, used in uncertainty principle contexts)
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 60x^10 + 1260x^8 - 11760x^6 + 45360x^4 - 60480x^2 + 10395
    
    # The function is g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # Constraint: g(0) = 0 => c0*1 + c4*3 + c8*105 + c12*10395 = 0
    # So c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    # We need to compute the upper bound on A(f)*A(F(f)).
    # For a Gaussian-weighted polynomial, the uncertainty product can be computed analytically.
    # The variance of x (position) and the variance of the Fourier transform (momentum) can be computed.
    # 
    # Let's define the probability density proportional to |g(x)|^2 * exp(-x^2).
    # Actually, the standard setup is f(x) = g(x) * exp(-x^2/2) or similar.
    # Let's assume f(x) = g(x) * exp(-x^2/2) so that the Gaussian part is standard normal.
    # Then |f(x)|^2 = |g(x)|^2 * exp(-x^2).
    # The normalization constant is Z = integral |g(x)|^2 exp(-x^2) dx.
    # The variance of x is <x^2> - <x>^2. By symmetry, <x> = 0.
    # The Fourier transform of exp(-x^2/2) is sqrt(2*pi)*exp(-k^2/2).
    # The Fourier transform of g(x)*exp(-x^2/2) is the convolution of the Fourier transform of g(x) with exp(-k^2/2).
    # This is getting complex. Let's use the known result for the uncertainty product for Hermite-Gaussian functions.
    # 
    # Alternative approach: Use the fact that for f(x) = g(x) exp(-x^2/2),
    # A(f) = sqrt(<x^2> - <x>^2) and A(F(f)) = sqrt(<k^2> - <k>^2) where the expectation is with respect to |f|^2 and |F(f)|^2.
    #
    # Let's compute <x^2> and <k^2> analytically.
    # Since g(x) is a polynomial, we can write g(x) = sum_{n=0}^{12} a_n x^n.
    # Then |g(x)|^2 = (sum a_n x^n)(sum a_m x^m) = sum_{n,m} a_n a_m x^{n+m}.
    # <x^2> = integral x^2 |g(x)|^2 exp(-x^2) dx / integral |g(x)|^2 exp(-x^2) dx.
    # integral x^{2k} exp(-x^2) dx = (2k-1)!! * sqrt(pi) / 2^k for k >= 0, with (-1)!! = 1.
    #
    # For the momentum space, F(f)(k) = integral f(x) exp(-ikx) dx.
    # f(x) = g(x) exp(-x^2/2).
    # F(f)(k) = sum_{n} a_n integral x^n exp(-x^2/2) exp(-ikx) dx.
    # integral x^n exp(-x^2/2 - ikx) dx = i^n H_n(k) exp(-k^2/2) ? No, that's for exp(-x^2/2) * exp(-ikx).
    # Actually, integral x^n exp(-x^2/2 - ikx) dx = i^n H_n(k) exp(-k^2/2) is not quite right.
    # The correct identity is: integral x^n exp(-x^2/2 - ikx) dx = i^n H_n(k) exp(-k^2/2) * sqrt(2*pi)?
    # Let's check for n=0: integral exp(-x^2/2 - ikx) dx = sqrt(2*pi) exp(-k^2/2).
    # For n=1: integral x exp(-x^2/2 - ikx) dx = i * d/dk [integral exp(-x^2/2 - ikx) dx] = i * sqrt(2*pi) * (-k) exp(-k^2/2) = -i k sqrt(2*pi) exp(-k^2/2).
    # H_1(k) = 2k. So i^1 H_1(k) exp(-k^2/2) = i * 2k * exp(-k^2/2). This doesn't match.
    # The correct relation is: integral x^n exp(-x^2/2 - ikx) dx = i^n H_n^{(phys)}(k) exp(-k^2/2) * sqrt(2*pi) is incorrect.
    # Actually, the Fourier transform of x^n exp(-x^2/2) is i^n H_n(k) exp(-k^2/2) where H_n are physicists' Hermite polynomials, but with a factor.
    # Let's use the property that the uncertainty product for a function of the form P(x) exp(-x^2/2) can be computed by evaluating the moments.
    #
    # Given the complexity, let's use a numerical integration approach for the objective function, but make it efficient.
    # Or, let's use the known analytical result for the variance of the momentum.
    # For f(x) = g(x) exp(-x^2/2), the momentum variance is <p^2> = integral |f'(x)|^2 dx (since p = -i d/dx, and for real functions, <p^2> = integral (f')^2 dx).
    # Wait, <p^2> = integral |(-i d/dx f(x))|^2 dx = integral |f'(x)|^2 dx.
    # So A(F(f)) = sqrt(integral |g'(x) exp(-x^2/2)|^2 dx / integral |g(x) exp(-x^2/2)|^2 dx) = sqrt(integral |g'(x)|^2 exp(-x^2) dx / integral |g(x)|^2 exp(-x^2) dx).
    #
    # So both A(f) and A(F(f)) can be computed from integrals of polynomials times exp(-x^2).
    # A(f) = sqrt(<x^2>), where <x^2> = integral x^2 |g(x)|^2 exp(-x^2) dx / integral |g(x)|^2 exp(-x^2) dx.
    # A(F(f)) = sqrt(<p^2>), where <p^2> = integral |g'(x)|^2 exp(-x^2) dx / integral |g(x)|^2 exp(-x^2) dx.
    #
    # Let's implement this analytically.
    # g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    def hermite_coeffs(deg):
        """Return coefficients of H_deg(x) in standard polynomial form [a0, a1, ..., a_deg]."""
        # Use recurrence or known values.
        # H0 = 1
        # H1 = 2x
        # H2 = 4x^2 - 2
        # H3 = 8x^3 - 12x
        # H4 = 16x^4 - 48x^2 + 12
        # H5 = 32x^5 - 160x^3 + 120x
        # H6 = 64x^6 - 480x^4 + 720x^2 - 120
        # H7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
        # H8 = 256x^8 - 4480x^6 + 20160x^4 - 20160x^2 + 1680
        # H9 = 512x^9 - 12288x^7 + 64512x^5 - 120960x^3 + 60480x
        # H10 = 1024x^10 - 32768x^8 + 241920x^6 - 580608x^4 + 453600x^2 - 100800
        # H11 = 2048x^11 - 81920x^9 + 829440x^7 - 3072000x^5 + 4084080x^3 - 1663200x
        # H12 = 4096x^12 - 204800x^10 + 2745600x^8 - 13977600x^6 + 29030400x^4 - 19958400x^2 + 302400
        
        if deg == 0:
            return np.array([1.0])
        elif deg == 1:
            return np.array([0.0, 2.0])
        elif deg == 2:
            return np.array([-2.0, 0.0, 4.0])
        elif deg == 3:
            return np.array([0.0, -12.0, 0.0, 8.0])
        elif deg == 4:
            return np.array([12.0, 0.0, -48.0, 0.0, 16.0])
        elif deg == 5:
            return np.array([0.0, 120.0, 0.0, -160.0, 0.0, 32.0])
        elif deg == 6:
            return np.array([-120.0, 0.0, 720.0, 0.0, -480.0, 0.0, 64.0])
        elif deg == 7:
            return np.array([0.0, -1680.0, 0.0, 3360.0, 0.0, -1344.0, 0.0, 128.0])
        elif deg == 8:
            return np.array([1680.0, 0.0, -20160.0, 0.0, 20160.0, 0.0, -4480.0, 0.0, 256.0])
        elif deg == 9:
            return np.array([0.0, 60480.0, 0.0, -120960.0, 0.0, 64512.0, 0.0, -12288.0, 0.0, 512.0])
        elif deg == 10:
            return np.array([-100800.0, 0.0, 453600.0, 0.0, -580608.0, 0.0, 241920.0, 0.0, -32768.0, 0.0, 1024.0])
        elif deg == 11:
            return np.array([0.0, -1663200.0, 0.0, 4084080.0, 0.0, -3072000.0, 0.0, 829440.0, 0.0, -81920.0, 0.0, 2048.0])
        elif deg == 12:
            return np.array([302400.0, 0.0, -19958400.0, 0.0, 29030400.0, 0.0, -13977600.0, 0.0, 2745600.0, 0.0, -204800.0, 0.0, 4096.0])
        else:
            raise ValueError("Degree not implemented")
    
    h0 = hermite_coeffs(0)
    h4 = hermite_coeffs(4)
    h8 = hermite_coeffs(8)
    h12 = hermite_coeffs(12)
    
    # Combine coefficients for g(x)
    # g(x) = c0*h0 + c4*h4 + c8*h8 + c12*h12
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    def objective(c0, c4, c8):
        """Compute the uncertainty product A(f)*A(F(f))."""
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        
        # Construct g(x) coefficients
        # Max degree is 12
        g_coeffs = np.zeros(13)
        g_coeffs[:len(h0)] += c0 * h0
        g_coeffs[:len(h4)] += c4 * h4
        g_coeffs[:len(h8)] += c8 * h8
        g_coeffs[:len(h12)] += c12 * h12
        
        # Compute |g(x)|^2 coefficients
        # g(x) = sum_{n=0}^{12} g_n x^n
        # |g(x)|^2 = g(x)^2 (since g is real)
        # g(x)^2 = sum_{k=0}^{24} b_k x^k
        b = np.convolve(g_coeffs, g_coeffs)
        
        # Compute integral of x^k exp(-x^2) dx from -inf to inf
        # integral x^k exp(-x^2) dx = 0 if k is odd
        # If k = 2m, integral = (2m-1)!! * sqrt(pi) / 2^m
        def moment_integral(k):
            if k % 2 != 0:
                return 0.0
            m = k // 2
            if m == 0:
                return np.sqrt(np.pi)
            # (2m-1)!! = (2m)! / (2^m * m!)
            result = factorial(2*m) / (2**m * factorial(m)) * np.sqrt(np.pi)
            return result
        
        # Compute Z = integral |g(x)|^2 exp(-x^2) dx = sum_k b_k * moment_integral(k)
        Z = 0.0
        for k in range(len(b)):
            Z += b[k] * moment_integral(k)
        
        if abs(Z) < 1e-12:
            return np.inf
        
        # Compute <x^2> = integral x^2 |g(x)|^2 exp(-x^2) dx / Z
        # integral x^2 |g(x)|^2 exp(-x^2) dx = sum_k b_k * moment_integral(k+2)
        x2_num = 0.0
        for k in range(len(b)):
            x2_num += b[k] * moment_integral(k+2)
        
        x2 = x2_num / Z
        
        # Compute g'(x)
        # g'(x) = sum_{n=1}^{12} n * g_n x^{n-1}
        gp_coeffs = np.zeros(12)
        for n in range(1, 13):
            gp_coeffs[n-1] = n * g_coeffs[n]
        
        # Compute |g'(x)|^2 = g'(x)^2
        bp = np.convolve(gp_coeffs, gp_coeffs)
        
        # Compute integral |g'(x)|^2 exp(-x^2) dx = sum_k bp_k * moment_integral(k)
        p2_num = 0.0
        for k in range(len(bp)):
            p2_num += bp[k] * moment_integral(k)
        
        p2 = p2_num / Z
        
        # A(f) = sqrt(x2), A(F(f)) = sqrt(p2)
        # Product = sqrt(x2 * p2)
        if x2 < 0 or p2 < 0:
            return np.inf
        
        return np.sqrt(x2 * p2)
    
    # Initial best solution (from seed)
    c0_init, c4_init, c8_init = -113/100, 1/25, 1/3240
    best_c = np.array([c0_init, c4_init, c8_init])
    best_val = objective(*best_c)
    
    # Stochastic local search with adaptive step sizes
    np.random.seed(42)
    
    # Parameterize on the unit sphere? No, the coefficients are not naturally on a sphere.
    # But we can use a reparameterization. Let's try a simple random restart local search.
    # Or, use the fact that the objective is scale-invariant?
    # Let's check: if we scale c0, c4, c8 by a factor s, then c12 also scales by s.
    # Then g(x) scales by s, so |g(x)|^2 scales by s^2.
    # Z scales by s^2, x2_num scales by s^2, p2_num scales by s^2.
    # So x2 and p2 are invariant under scaling.
    # Thus, the objective is scale-invariant. We can fix one coefficient, e.g., c0 = 1, or normalize the vector.
    # Let's normalize so that c0^2 + c4^2 + c8^2 = 1.
    
    def normalize(c):
        norm = np.linalg.norm(c)
        if norm < 1e-12:
            return c
        return c / norm
    
    # Initial normalized vector
    c_init = np.array([c0_init, c4_init, c8_init])
    c_init = normalize(c_init)
    best_c_norm = c_init.copy()
    best_val = objective(*best_c_norm)
    
    # Local search
    step_size = 0.1
    n_iter = 5000
    patience = 100
    no_improve = 0
    
    for i in range(n_iter):
        if time.time() > deadline:
            break
        
        # Generate a random perturbation
        delta = np.random.randn(3) * step_size
        new_c = best_c_norm + delta
        new_c = normalize(new_c)
        
        new_val = objective(*new_c)
        
        if new_val < best_val:
            best_val = new_val
            best_c_norm = new_c
            no_improve = 0
            # Adaptive step size: increase slightly
            step_size = min(step_size * 1.05, 1.0)
        else:
            no_improve += 1
            # Adaptive step size: decrease if no improvement
            if no_improve > patience:
                step_size = max(step_size * 0.9, 1e-4)
                no_improve = 0
        
        # Random restart occasionally
        if i % 500 == 0 and i > 0:
            # Try a few random restarts
            for _ in range(5):
                restart_c = np.random.randn(3)
                restart_c = normalize(restart_c)
                restart_val = objective(*restart_c)
                if restart_val < best_val:
                    best_val = restart_val
                    best_c_norm = restart_c
    
    # Return the best coefficients
    return best_c_norm

# EVOLVE-BLOCK-END
