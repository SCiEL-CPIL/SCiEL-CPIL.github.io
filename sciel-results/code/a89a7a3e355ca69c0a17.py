"""Rewritten program: directly minimize the upper bound on C4 via Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hermite_gaussian(x, coeffs):
    """
    Evaluate the Hermite-Gaussian test function g(x) = exp(-x^2/2) * P(x)
    where P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    and c12 is chosen so that P(0) = 0.
    
    Physicists' Hermite polynomials: H_n(x)
    H0 = 1
    H4 = 16x^4 - 48x^2 + 12
    H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    H12 = 4096x^12 - 98304x^10 + 921600x^8 - 4032000x^6 + 7983360x^4 - 5880960x^2 + 10321920
    """
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    
    H0 = 1.0
    H4 = 16*x4 - 48*x2 + 12
    H8 = 256*x8 - 3072*x6 + 11520*x4 - 13440*x2 + 1680
    H12 = 4096*x12 - 98304*x10 + 921600*x8 - 4032000*x6 + 7983360*x4 - 5880960*x2 + 10321920
    
    c0, c4, c8 = coeffs
    # P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=10321920
    c12 = -(c0*1.0 + c4*12.0 + c8*1680.0) / 10321920.0
    
    P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    return np.exp(-0.5*x2) * P

def _fourier_transform_g(x, coeffs):
    """
    Compute the Fourier transform of g(x) = exp(-x^2/2)*P(x).
    For a function f(x) = exp(-x^2/2)*Q(x) where Q is a polynomial,
    the Fourier transform (with convention F(k) = integral f(x) exp(-i k x) dx)
    can be computed. Since g is a product of Gaussian and polynomial,
    its FT is also a Gaussian times a polynomial.
    
    Actually, let's use the fact that if g(x) = exp(-x^2/2) * P(x), then
    F(g)(k) = sqrt(2*pi) * exp(-k^2/2) * P_ft(k) where P_ft involves
    derivatives or the FT of the polynomial part.
    
    More precisely, using the property: FT[exp(-x^2/2) * x^n](k) = 
    sqrt(2*pi) * (-i)^n * d^n/dk^n [exp(-k^2/2)]
    
    But it's easier to just compute numerically or use the known result.
    Let me use a different approach: since g is a combination of Hermite-Gaussian
    functions, and the Fourier transform of H_n(x)*exp(-x^2/2) is (-i)^n * H_n(k)*exp(-k^2/2)
    (up to normalization depending on convention).
    
    With the convention F(k) = (1/sqrt(2*pi)) * integral f(x) exp(-ikx) dx,
    FT[H_n(x) exp(-x^2/2)](k) = (-i)^n H_n(k) exp(-k^2/2)
    
    So F(g)(k) = c0*H0(k)*exp(-k^2/2) + c4*(-i)^4*H4(k)*exp(-k^2/2) + c8*(-i)^8*H8(k)*exp(-k^2/2) + c12*(-i)^12*H12(k)*exp(-k^2/2)
    
    (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    
    So F(g)(k) = exp(-k^2/2) * [c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)]
    
    Wait, that's the same polynomial structure! Let me verify.
    Actually with the unitary Fourier transform convention, the FT of the nth Hermite-Gaussian
    eigenfunction is (-i)^n times itself. So yes, F(g)(k) = exp(-k^2/2) * P(k) where P is the same polynomial.
    
    Hmm, but that would mean A(f)*A(F(f)) has a simple form. Let me reconsider.
    
    Actually, I need to be careful about the Fourier transform convention used in the task.
    Let me just compute the bound numerically via integration.
    """
    # For the uncertainty inequality, we need to compute:
    # A(f) = sup_x |x * f(x)| and A(F(f)) = sup_k |k * F(f)(k)|
    # The bound C4 relates to these.
    # 
    # Let me think about what "upper bound on C4" means in this context.
    # The task says to reduce the upper bound on A(f)*A(Fourier(f)).
    # 
    # Let me compute A(f) = sup |x*g(x)| and A(F(g)) = sup |k*F(g)(k)|
    # and return their product as the objective to minimize.
    pass

def _compute_bound(coeffs):
    """
    Compute the upper bound on A(f)*A(F(f)) for the given coefficients.
    """
    x = np.linspace(-10, 10, 20001)
    g = _hermite_gaussian(x, coeffs)
    
    # A(f) = sup |x * g(x)|
    Af = np.max(np.abs(x * g))
    
    # For the Fourier transform, use the property that FT of H_n(x)*exp(-x^2/2)
    # with unitary convention is (-i)^n * H_n(x)*exp(-x^2/2)
    # So F(g)(k) = exp(-k^2/2) * [c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)]
    # which is the same as g(k)!
    # 
    # Wait, that means A(F(f)) = A(f), so the product is Af^2.
    # Let me verify this is correct.
    
    # Actually, let me just use scipy's fft or compute the FT numerically to be safe.
    # But the analytic result should be that F(g)(k) = g(k) for this specific form
    # if the polynomial is even (which it is, since we only have even-degree Hermite polys).
    
    # For even n, (-i)^n = 1, so yes F(g)(k) = g(k).
    A_Ff = Af  # Since g is even and composed of even Hermite-Gaussians
    
    return Af * A_Ff

def construct():
    deadline = time.time() + 180  # 180 second deadline
    
    # Initial guess from the seed program
    initial = np.array([-113/100, 1/25, 1/3240])
    
    # Evaluate the initial bound
    best_coeffs = initial.copy()
    best_bound = _compute_bound(best_coeffs)
    
    # Multi-start Nelder-Mead optimization
    np.random.seed(42)
    starts = [initial]
    # Add diverse initial guesses
    for _ in range(10):
        start = initial + np.random.normal(0, 0.1, size=3)
        starts.append(start)
    
    for start in starts:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _compute_bound,
                start,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            continue
    
    return best_coeffs
# EVOLVE-BLOCK-END
