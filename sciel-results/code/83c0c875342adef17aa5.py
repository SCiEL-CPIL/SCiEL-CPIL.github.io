"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Returns three coefficients for the Hermite combination H0, H4, H8.
    The degree-12 weight is fixed so that g(0) = 0.
    
    We minimize the uncertainty product upper bound C4.
    """
    # Deadline guard: evaluator kills after 210s, we want to be safe at ~180s
    start_time = time.time()
    DEADLINE = 180.0
    
    # The test function is g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # with constraint g(0) = 0.
    # Hermite polynomials (physicists' convention, as typically used in uncertainty problems):
    # H0(x) = 1
    # H1(x) = 2x
    # H2(x) = 4x^2 - 2
    # H3(x) = 8x^3 - 12x
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12(x) = 4096x^12 - 76800x^10 + 645120x^8 - 2611200x^6 + 5080320x^4 - 4032000x^2 + 1039500
    
    # Wait, let me verify the standard Hermite polynomials.
    # Physicists' Hermite polynomials H_n(x):
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 4x^2 - 2
    # H_3 = 8x^3 - 12x
    # H_4 = 16x^4 - 48x^2 + 12
    # H_5 = 32x^5 - 320x^3 + 480x
    # H_6 = 64x^6 - 1920x^4 + 11520x^2 - 1680
    # H_7 = 128x^7 - 8960x^5 + 67200x^3 - 13440x
    # H_8 = 256x^8 - 30720x^6 + 215040x^4 - 358400x^2 + 16800
    
    # Let me double-check H_8:
    # H_8(x) = 256x^8 - 30720x^6 + 215040x^4 - 358400x^2 + 16800
    # H_8(0) = 16800
    
    # H_12(x) = 4096x^12 - 76800x^10 + 645120x^8 - 2611200x^6 + 5080320x^4 - 4032000x^2 + 1039500
    # H_12(0) = 1039500
    
    # Actually, let me be more careful. The standard recurrence is:
    # H_0 = 1, H_1 = 2x, H_{n+1} = 2x H_n - 2n H_{n-1}
    
    # H_2 = 2x(2x) - 2(1)(1) = 4x^2 - 2
    # H_3 = 2x(4x^2 - 2) - 4(2x) = 8x^3 - 4x - 8x = 8x^3 - 12x
    # H_4 = 2x(8x^3 - 12x) - 8(4x^2 - 2) = 16x^4 - 24x^2 - 32x^2 + 16 = 16x^4 - 56x^2 + 16
    # Wait, that doesn't match. Let me recalculate.
    # H_4 = 2x H_3 - 2*3 H_2 = 2x(8x^3 - 12x) - 6(4x^2 - 2) = 16x^4 - 24x^2 - 24x^2 + 12 = 16x^4 - 48x^2 + 12
    # Yes, that matches.
    
    # H_5 = 2x H_4 - 2*4 H_3 = 2x(16x^4 - 48x^2 + 12) - 8(8x^3 - 12x) = 32x^5 - 96x^3 + 24x - 64x^3 + 96x = 32x^5 - 160x^3 + 120x
    # H_6 = 2x H_5 - 2*5 H_4 = 2x(32x^5 - 160x^3 + 120x) - 10(16x^4 - 48x^2 + 12) = 64x^6 - 320x^4 + 240x^2 - 160x^4 + 480x^2 - 120 = 64x^6 - 480x^4 + 720x^2 - 120
    # H_7 = 2x H_6 - 2*6 H_5 = 2x(64x^6 - 480x^4 + 720x^2 - 120) - 12(32x^5 - 160x^3 + 120x) = 128x^7 - 960x^5 + 1440x^3 - 240x - 384x^5 + 1920x^3 - 1440x = 128x^7 - 1344x^5 + 3360x^3 - 1680x
    # H_8 = 2x H_7 - 2*7 H_6 = 2x(128x^7 - 1344x^5 + 3360x^3 - 1680x) - 14(64x^6 - 480x^4 + 720x^2 - 120) = 256x^8 - 2688x^6 + 6720x^4 - 3360x^2 - 896x^6 + 6720x^4 - 10080x^2 + 1680 = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
    
    # So H_8(0) = 1680.
    
    # H_9 = 2x H_8 - 2*8 H_7 = 2x(256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680) - 16(128x^7 - 1344x^5 + 3360x^3 - 1680x)
    # = 512x^9 - 7168x^7 + 26880x^5 - 26880x^3 + 3360x - 2048x^7 + 21504x^5 - 53760x^3 + 26880x
    # = 512x^9 - 9216x^7 + 48384x^5 - 80640x^3 + 30240x
    
    # H_10 = 2x H_9 - 2*9 H_8 = 2x(512x^9 - 9216x^7 + 48384x^5 - 80640x^3 + 30240x) - 18(256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680)
    # = 1024x^10 - 18432x^8 + 96768x^6 - 161280x^4 + 60480x^2 - 4608x^8 + 64512x^6 - 241920x^4 + 241920x^2 - 30240
    # = 1024x^10 - 23040x^8 + 161280x^6 - 403200x^4 + 302400x^2 - 30240
    
    # H_11 = 2x H_10 - 2*10 H_9 = 2x(1024x^10 - 23040x^8 + 161280x^6 - 403200x^4 + 302400x^2 - 30240) - 20(512x^9 - 9216x^7 + 48384x^5 - 80640x^3 + 30240x)
    # = 2048x^11 - 46080x^9 + 322560x^7 - 806400x^5 + 604800x^3 - 60480x - 10240x^9 + 184320x^7 - 967680x^5 + 1612800x^3 - 604800x
    # = 2048x^11 - 56320x^9 + 506880x^7 - 1774080x^5 + 2217600x^3 - 665280x
    
    # H_12 = 2x H_11 - 2*11 H_10 = 2x(2048x^11 - 56320x^9 + 506880x^7 - 1774080x^5 + 2217600x^3 - 665280x) - 22(1024x^10 - 23040x^8 + 161280x^6 - 403200x^4 + 302400x^2 - 30240)
    # = 4096x^12 - 112640x^10 + 1013760x^8 - 3548160x^6 + 4435200x^4 - 1330560x^2 - 22528x^10 + 506880x^8 - 3548160x^6 + 8870400x^4 - 6652800x^2 + 665280
    # = 4096x^12 - 135168x^10 + 1520640x^8 - 7096320x^6 + 13305600x^4 - 7983360x^2 + 665280
    
    # So H_12(0) = 665280.
    
    # Let me verify H_12(0) more carefully.
    # H_12 = 2x H_11 - 22 H_10
    # H_11(0) = 0 (odd polynomial)
    # H_10(0) = -30240
    # So H_12(0) = 22 * 30240 = 665280. Yes.
    
    # H_8(0) = 1680
    # H_4(0) = 12
    # H_0(0) = 1
    
    # Constraint: g(0) = c0*H_0(0) + c4*H_4(0) + c8*H_8(0) + c12*H_12(0) = 0
    # c0 * 1 + c4 * 12 + c8 * 1680 + c12 * 665280 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 665280
    
    # Now, the uncertainty product. For a function g(x), the uncertainty in position is:
    # <x^2> = integral x^2 |g(x)|^2 dx / integral |g(x)|^2 dx
    # The uncertainty in momentum (Fourier space) is:
    # <p^2> = integral p^2 |G(p)|^2 dp / integral |G(p)|^2 dp
    # where G is the Fourier transform of g.
    
    # For Hermite-Gaussian functions, there are known formulas. But here we have a linear combination.
    
    # Let me think about this differently. The problem says "Fourier-invariant Hermite-polynomial times Gaussian test function".
    # So g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2) or similar.
    
    # Actually, the standard Hermite-Gaussian functions are psi_n(x) = H_n(x) * exp(-x^2/2) / normalization.
    
    # Let's assume g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2).
    
    # The Fourier transform of H_n(x) exp(-x^2/2) is related to H_n(k) exp(-k^2/2) with a phase factor.
    # Specifically, for the physicists' convention, the Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(k) e^{-k^2/2} (up to normalization).
    
    # So the momentum-space wavefunction is a linear combination of the same Hermite-Gaussian functions with coefficients (-i)^n c_n.
    
    # For even n, (-i)^n = (-1)^{n/2}.
    # n=0: (-i)^0 = 1
    # n=4: (-i)^4 = 1
    # n=8: (-i)^8 = 1
    # n=12: (-i)^12 = 1
    
    # So all coefficients are unchanged in momentum space (up to an overall phase).
    
    # This means the position and momentum variances are computed from the same linear combination.
    
    # Let's define f(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    
    # We need to compute:
    # <x^2> = integral x^2 |f(x)|^2 dx / integral |f(x)|^2 dx
    # <p^2> = integral p^2 |F(p)|^2 dp / integral |F(p)|^2 dp
    
    # Since the Fourier transform preserves the L2 norm (up to a constant), and the form is similar,
    # <p^2> will have the same form as <x^2> but with the Fourier-transformed coefficients.
    
    # For even Hermite polynomials, the Fourier transform multiplies by (-i)^n = (-1)^{n/2}.
    # All our n are divisible by 4, so (-1)^{n/2} = 1 for n=0,4,8,12.
    
    # Wait, n=0: (-1)^0 = 1
    # n=4: (-1)^2 = 1
    # n=8: (-1)^4 = 1
    # n=12: (-1)^6 = 1
    
    # So the coefficients are the same in position and momentum space!
    
    # This means <x^2> = <p^2> for this particular combination? That can't be right for a general combination.
    
    # Let me reconsider. The Fourier transform of x^n e^{-x^2/2} involves derivatives of the delta function or related functions.
    
    # Actually, let's use the property that the Hermite-Gaussian functions are eigenfunctions of the Fourier transform.
    # psi_n(x) = H_n(x) e^{-x^2/2}
    # F[psi_n](k) = (-i)^n psi_n(k)
    
    # So if f(x) = sum c_n psi_n(x), then F[f](k) = sum c_n (-i)^n psi_n(k).
    
    # For our case, all n are even and divisible by 4, so (-i)^n = 1.
    # Thus F[f](k) = sum c_n psi_n(k) = f(k).
    
    # So f is an eigenfunction of the Fourier transform with eigenvalue 1!
    
    # This means <x^2> = <p^2>.
    
    # If that's the case, the uncertainty product is <x^2> * <p^2> = <x^2>^2.
    
    # And we want to minimize this, subject to the normalization and the constraint g(0)=0.
    
    # Wait, but the problem says "reduce the upper bound on A(f)*A(Fourier(f))". If A(f) is the standard deviation, then A(f)*A(F(f)) = sqrt(<x^2>) * sqrt(<p^2>) = <x^2> if <x^2>=<p^2>.
    
    # So we want to minimize <x^2> subject to the constraints.
    
    # Let's compute <x^2> = integral x^2 |f(x)|^2 dx / integral |f(x)|^2 dx.
    
    # f(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) e^{-x^2/2}
    
    # |f(x)|^2 = (c0*H0 + c4*H4 + c8*H8 + c12*H12)^2 e^{-x^2}
    
    # Let P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    # Then <x^2> = integral x^2 P(x)^2 e^{-x^2} dx / integral P(x)^2 e^{-x^2} dx
    
    # This is a ratio of integrals. We can compute these analytically or numerically.
    
    # Let's use numerical integration for now, and then optimize.
    
    from scipy.integrate import quad
    
    H0 = lambda x: np.ones_like(x)
    H4 = lambda x: 16*x**4 - 48*x**2 + 12
    H8 = lambda x: 256*x**8 - 3584*x**6 + 13440*x**4 - 13440*x**2 + 1680
    H12 = lambda x: 4096*x**12 - 135168*x**10 + 1520640*x**8 - 7096320*x**6 + 13305600*x**4 - 7983360*x**2 + 665280
    
    def P(x, c0, c4, c8):
        c12 = -(c0 + 12*c4 + 1680*c8) / 665280
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def objective(c):
        c0, c4, c8 = c
        c12 = -(c0 + 12*c4 + 1680*c8) / 665280
        if c0 == 0 and c4 == 0 and c8 == 0:
            return np.inf
        # Compute <x^2>
        # integrand for numerator: x^2 * P(x)^2 * exp(-x^2)
        # integrand for denominator: P(x)^2 * exp(-x^2)
        
        def num_integrand(x):
            px = P(np.array([x]), c0, c4, c8)[0]
            return x**2 * px**2 * np.exp(-x**2)
        
        def den_integrand(x):
            px = P(np.array([x]), c0, c4, c8)[0]
            return px**2 * np.exp(-x**2)
        
        # Integrate from -inf to inf
        # Use quad for numerical integration
        num, _ = quad(num_integrand, -np.inf, np.inf, limit=100)
        den, _ = quad(den_integrand, -np.inf, np.inf, limit=100)
        
        if den <= 0:
            return np.inf
        
        x2 = num / den
        # The uncertainty product is sqrt(x2) * sqrt(x2) = x2 (if <x^2>=<p^2>)
        # Or is it x2 * x2 = x2^2?
        # A(f) = sqrt(<x^2>), A(F(f)) = sqrt(<p^2>)
        # Product = sqrt(<x^2> * <p^2>) = <x^2> if <x^2>=<p^2>
        return x2
    
    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try to minimize
    best_result = None
    best_val = np.inf
    
    # Use multiple starting points
    start_points = [x0,
                    np.array([-1.0, 0.0, 0.0]),
                    np.array([0.0, -1.0, 0.0]),
                    np.array([0.0, 0.0, -1.0]),
                    np.array([1.0, 1.0, 1.0]),
                    np.array([-1.0, 1.0, -1.0])]
    
    for sp in start_points:
        if time.time() - start_time > DEADLINE - 30:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is not None:
        c0, c4, c8 = best_result
        return np.array([c0, c4, c8])
    else:
        return x0

# EVOLVE-BLOCK-END
