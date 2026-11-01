"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def construct():
    # We need to find coefficients a0, a4, a8 for the test function
    # g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    # where a12 is chosen such that g(0) = 0.
    #
    # Hermite polynomials (physicist's convention):
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 6144x^6 + 40320x^4 - 80640x^2 + 30240
    # H12(x) = 4096x^12 - 196608x^10 + 2949120x^8 - 16257024x^6 + 40840704x^4 - 47185920x^2 + 13063680
    #
    # At x=0:
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 30240
    # H12(0) = 13063680
    #
    # g(0) = (a0 + 12*a4 + 30240*a8 + a12*13063680) * 1 = 0
    # => a12 = -(a0 + 12*a4 + 30240*a8) / 13063680
    #
    # The Fourier transform of H_n(x)*exp(-x^2/2) is i^n * H_n(x) * exp(-x^2/2)
    # (up to a normalization constant depending on the Fourier transform convention)
    #
    # For the uncertainty principle bound A(f)*A(F(f)), we need to compute
    # the L1 norm (or some norm) of f and its Fourier transform.
    #
    # Let's think about this more carefully. The task says "upper bound on A(f)*A(Fourier(f))"
    # where A is likely the L1 norm or some moment-based quantity.
    #
    # Given the structure, let's assume we're minimizing a bound that depends on
    # the coefficients. The seed program returns [-1.13, 0.04, 1/3240] which is the
    # state of the art from Goncalves et al. 2017.
    #
    # Let me set up the optimization. The key insight is that for Hermite-Gaussian
    # functions, the Fourier transform has a simple form. Let's define:
    # g(x) = sum_{k in {0,4,8,12}} a_k * H_k(x) * exp(-x^2/2)
    # 
    # The Fourier transform (with convention F(f)(xi) = int f(x) e^{-i x xi} dx)
    # of H_n(x) exp(-x^2/2) is i^n sqrt(2*pi) H_n(x) exp(-x^2/2)
    #
    # So F(g)(xi) = sqrt(2*pi) * sum_{k} a_k * i^k * H_k(xi) * exp(-xi^2/2)
    #
    # For the uncertainty bound, we likely need ||g||_1 * ||F(g)||_1 or similar.
    #
    # Let's compute ||g||_1 = |int g(x) dx| and ||F(g)||_1 = |int F(g)(xi) dxi|
    #
    # int H_n(x) exp(-x^2/2) dx = 0 for n > 0, and sqrt(2*pi) for n=0
    #
    # So int g(x) dx = a0 * sqrt(2*pi)
    #
    # And int F(g)(xi) dxi = F(g)(0) * 2*pi = sqrt(2*pi) * 2*pi * sum a_k * i^k * H_k(0) * exp(0)
    # Wait, let me be more careful.
    #
    # Actually, let me reconsider. The "upper bound" might refer to a specific
    # inequality bound. Let me look at what makes sense.
    #
    # A common formulation: if f is in L1 and F(f) is in L1, then
    # ||f||_1 * ||F(f)||_1 >= C (some constant)
    #
    # The task is to find g that minimizes this product, subject to g(0)=0.
    #
    # Let's compute:
    # ||g||_1 = |int g(x) dx| = |a0| * sqrt(2*pi)
    #
    # For ||F(g)||_1, we need |int F(g)(xi) dxi|
    # int F(g)(xi) dxi = 2*pi * g(0) = 0 (by our constraint!)
    #
    # Hmm, that's zero. So maybe A is not the L1 norm but something else.
    #
    # Perhaps A(f) is the L2 norm? ||f||_2 * ||F(f)||_2 is constant by Plancherel.
    #
    # Or perhaps A(f) involves moments? Like sqrt(int x^2 |f(x)|^2 dx)?
    #
    # Let me reconsider. In uncertainty principle contexts, A(f) often means
    # the "width" or spread. For the Heisenberg uncertainty principle:
    # (int x^2 |f|^2 dx) * (int xi^2 |F(f)|^2 dxi) >= (1/4) (int |f|^2 dx)^2
    #
    # But the task says "upper bound on A(f)*A(Fourier(f))" which we want to reduce.
    #
    # Given the seed values and the structure, let me try a numerical approach:
    # define g from the coefficients, compute some reasonable measure, and optimize.
    #
    # Let me try: A(f) = sqrt(int x^2 |f(x)|^2 dx / int |f(x)|^2 dx) (standard deviation)
    # and minimize the product of this for g and F(g).
    #
    # But actually, for Hermite functions, the Fourier transform is well-behaved.
    # Let me just set up a numerical optimization.
    
    deadline = time.time() + 190  # Safety margin from 210s limit
    
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return 16*x**4 - 48*x**2 + 12
    
    def H8(x):
        return 256*x**8 - 6144*x**6 + 40320*x**4 - 80640*x**2 + 30240
    
    def H12(x):
        return (4096*x**12 - 196608*x**10 + 2949120*x**8 - 
                16257024*x**6 + 40840704*x**4 - 47185920*x**2 + 13063680)
    
    def compute_bound(a0, a4, a8):
        """Compute the upper bound on A(g)*A(F(g)) for given coefficients."""
        a12 = -(a0 + 12*a4 + 30240*a8) / 13063680.0
        
        # Use a grid for numerical integration
        x = np.linspace(-15, 15, 20000)
        dx = x[1] - x[0]
        gauss = np.exp(-x**2 / 2)
        
        g = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * gauss
        
        # Compute A(g) as sqrt(int x^2 |g|^2 dx / int |g|^2 dx)
        norm_sq = np.trapz(g**2, x)
        if norm_sq < 1e-15:
            return 1e10  # Invalid
        
        var_x = np.trapz(x**2 * g**2, x) / norm_sq
        
        # Fourier transform of g
        # F(g)(xi) = int g(x) exp(-i x xi) dx
        # For efficiency, use FFT or direct computation
        # Since g is a sum of Hermite-Gaussians, F(g) is also a sum
        # F(H_n(x) exp(-x^2/2)) = i^n sqrt(2*pi) H_n(x) exp(-x^2/2)
        
        xi = x  # Use same grid for frequency
        Fg = (a0*H0(xi) + a4*(1j)**4*H4(xi) + a8*(1j)**8*H8(xi) + 
              a12*(1j)**12*H12(xi)) * np.sqrt(2*np.pi) * np.exp(-xi**2/2)
        
        norm_sq_F = np.trapz(np.abs(Fg)**2, xi)
        if norm_sq_F < 1e-15:
            return 1e10
        
        var_xi = np.trapz(xi**2 * np.abs(Fg)**2, xi) / norm_sq_F
        
        A_g = np.sqrt(var_x)
        A_Fg = np.sqrt(var_xi)
        
        return A_g * A_Fg
    
    # Start with the seed values
    seed = np.array([-113/100, 1/25, 1/3240])
    best_val = compute_bound(*seed)
    best_x = seed.copy()
    
    # Try local optimization from the seed
    try:
        result = minimize(
            lambda x: compute_bound(x[0], x[1], x[2]),
            seed,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_x
    
    # Try differential evolution for global search
    try:
        bounds = [(-5, 5), (-1, 1), (-0.1, 0.1)]
        de_result = differential_evolution(
            lambda x: compute_bound(x[0], x[1], x[2]),
            bounds,
            maxiter=200,
            popsize=15,
            seed=42,
            tol=1e-8
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_x = de_result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return best_x
    
    # Final polish
    try:
        result = minimize(
            lambda x: compute_bound(x[0], x[1], x[2]),
            best_x,
            method='Powell',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x

# EVOLVE-BLOCK-END
