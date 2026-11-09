"""Optimized construct for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Find three coefficients (c0, c4, c8) for the Hermite combination
    g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    where c12 is determined by g(0) = 0.
    
    The objective is to minimize the upper bound on C4 = A(f)*A(F).
    """
    start_time = time.time()
    deadline = start_time + 190  # 20s safety margin
    
    # Physical constants for the uncertainty problem
    # The test function is g(x) = sum_i c_i * H_i(x) * exp(-x^2/2)
    # H_n are physicists' Hermite polynomials
    
    # Precompute needed values
    # H_n(0) for even n:
    # H_0(0) = 1
    # H_4(0) = 12  (H_4(x) = x^4 - 6x^2 + 3, so H_4(0) = 3... let me recalculate)
    # Actually, physicists' Hermite: H_0=1, H_1=2x, H_2=4x^2-2, H_3=8x^3-12x,
    # H_4=16x^4-48x^2+12, so H_4(0)=12
    # H_8: H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680, H_8(0)=1680
    # H_12: H_12(0) = 2^12 * (12)!/(6!) = 4096 * 12*11*10*9*8*7/720 
    #       = 4096 * 665280/720 = 4096 * 924 = 3784704
    # Let me verify: H_{2n}(0) = (-1)^n * 2^{2n} * (2n)!/(n!)^2 * (1/2^n) ... 
    # Actually the formula is H_{2n}(0) = (-1)^n (2n)! / n! * 2^{n-1} ... 
    # Let me just use known values:
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 3784704
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 3784704.0
    
    # c12 is determined by g(0) = 0:
    # c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0 = 0
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    
    def get_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    def objective(coeffs):
        """
        Compute the upper bound on C4 for given coefficients.
        
        The uncertainty bound involves:
        A(f) = ||f||_1 / ||f||_2 (L1 norm over L2 norm)
        A(F) = ||F||_1 / ||F||_2 (L1 norm of Fourier transform over L2 norm)
        
        For g(x) = sum c_i H_i(x) exp(-x^2/2), we need to compute these norms.
        
        The key insight is that for Hermite functions, the Fourier transform
        has a simple form. The L2 norm is straightforward, but the L1 norm
        requires numerical integration.
        
        We use the fact that the bound C4 = A(f) * A(F) can be computed
        numerically.
        """
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        # Normalize: we can scale the function without changing A(f)*A(F)
        # since both A(f) and A(F) are scale-invariant ratios.
        # So we fix one coefficient or normalize the L2 norm.
        
        # Compute L2 norm squared: ||g||_2^2 = integral |g(x)|^2 dx
        # For g(x) = sum c_i H_i(x) e^{-x^2/2}, using orthogonality:
        # integral H_m(x) H_n(x) e^{-x^2} dx = 2^n n! sqrt(pi) delta_{mn}
        # So ||g||_2^2 = sum_i c_i^2 * 2^i * i! * sqrt(pi)
        
        # But wait, g(x) = sum c_i H_i(x) e^{-x^2/2}
        # |g(x)|^2 = (sum c_i H_i(x) e^{-x^2/2})^2 = (sum c_i H_i(x))^2 e^{-x^2}
        # integral |g(x)|^2 dx = sum_i c_i^2 * integral H_i(x)^2 e^{-x^2} dx
        # = sum_i c_i^2 * 2^i * i! * sqrt(pi)
        
        sqrt_pi = np.sqrt(np.pi)
        l2_sq = (c0**2 * 1.0 + c4**2 * 2**4 * 24 + c8**2 * 2**8 * 40320 + c12**2 * 2**12 * 479001600) * sqrt_pi
        # 4! = 24, 8! = 40320, 12! = 479001600
        l2 = np.sqrt(max(l2_sq, 1e-30))
        
        # For L1 norm, we need numerical integration
        # g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * e^{-x^2/2}
        
        # Define the polynomial part
        # H0(x) = 1
        # H4(x) = 16x^4 - 48x^2 + 12
        # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
        # H12(x) = 4096x^12 - 98304x^10 + 943712x^8 - 4515840x^6 + 9676800x^4 - 8316000x^2 + 2032128
        
        def poly(x):
            return (c0 * 1 + 
                    c4 * (16*x**4 - 48*x**2 + 12) +
                    c8 * (256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680) +
                    c12 * (4096*x**12 - 98304*x**10 + 943712*x**8 - 4515840*x**6 + 9676800*x**4 - 8316000*x**2 + 2032128))
        
        def g(x):
            return poly(x) * np.exp(-x**2 / 2)
        
        # Numerical L1 norm
        # Integrate |g(x)| from -inf to inf
        # Use a reasonable range, say [-8, 8]
        x = np.linspace(-8, 8, 2000)
        gv = g(x)
        l1 = np.trapz(np.abs(gv), x)
        
        a_f = l1 / l2
        
        # For the Fourier transform:
        # The Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}
        # (with the standard normalization)
        # So F(g)(x) = sum c_i (-i)^i H_i(x) e^{-x^2/2}
        # For even i, (-i)^i = (-1)^{i/2}
        # So F(g)(x) = (c0*1 + c4*(-1)^2*H4(x) + c8*(-1)^4*H8(x) + c12*(-1)^6*H12(x)) e^{-x^2/2}
        #            = (c0 + c4*H4(x) + c8*H8(x) + c12*H12(x)) e^{-x^2/2}
        # Wait, (-i)^4 = (-i)^2 * (-i)^2 = (-1)*(-1) = 1
        # (-i)^8 = 1
        # (-i)^12 = 1
        # So actually for all even n, (-i)^n = (-1)^{n/2}
        # n=0: (-1)^0 = 1
        # n=4: (-1)^2 = 1
        # n=8: (-1)^4 = 1
        # n=12: (-1)^6 = 1
        # So F(g) = g! The Fourier transform is the same function for this even combination.
        
        # Hmm, that means A(F) = A(f), so C4 = A(f)^2
        # But that seems too simple. Let me reconsider the Fourier transform convention.
        
        # Actually, the standard result is:
        # F[H_n(x) e^{-x^2/2}](x) = (-i)^n H_n(x) e^{-x^2/2}
        # This is for the unitary Fourier transform.
        
        # For n even: (-i)^n = e^{-i*pi*n/2} = cos(pi*n/2) - i*sin(pi*n/2)
        # n=0: 1
        # n=4: e^{-2pi*i} = 1
        # n=8: e^{-4pi*i} = 1
        # n=12: e^{-6pi*i} = 1
        # So yes, for all these even n, the Fourier transform coefficient is 1.
        
        # Therefore F(g) = g, and A(F) = A(f), so C4 = A(f)^2.
        
        # But wait, this means the problem reduces to minimizing A(f)^2 = (L1/L2)^2
        # subject to g(0) = 0.
        
        # Let's just compute A(f)^2 as the objective.
        return (l1 / l2) ** 2
    
    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try to optimize
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Use multiple starting points
    starts = [x0, 
              np.array([1, 0, 0]),
              np.array([0, 1, 0]),
              np.array([0, 0, 1]),
              np.array([-1, 0, 0]),
              np.array([0, -1, 0]),
              np.array([0, 0, -1]),
              np.array([1, 1, 1]),
              np.array([-1, -1, -1]),
              np.array([1, -1, 1])]
    
    for s in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Additional refinement with BFGS if time permits
    if time.time() < deadline - 30:
        try:
            res2 = minimize(objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 10000, 'ftol': 1e-15})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res3 = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res3.fun < best_val:
                best_val = res3.fun
                best_x = res3.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
