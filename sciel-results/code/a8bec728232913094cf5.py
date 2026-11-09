"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(x):
    """
    Compute the upper bound C4 for the given coefficients.
    x = [c0, c4, c8]
    The degree-12 coefficient c12 is determined by g(0) = 0.
    """
    c0, c4, c8 = x
    
    # Hermite polynomials (probabilists' convention? No, physicists' convention is standard for this task usually)
    # Let's check the standard definition.
    # H_n(x) = (-1)^n e^{x^2} d^n/dx^n (e^{-x^2})
    # H_0(x) = 1
    # H_4(x) = 16x^4 - 48x^2 + 12
    # H_8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H_12(x) = 4096x^12 - 46080x^10 + 241920x^8 - 552960x^6 + 645120x^4 - 322560x^2 + 40320
    
    # g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # Condition g(0) = 0:
    # c0*1 + c4*12 + c8*168 + c12*40320 = 0
    # c12 = -(c0 + 12*c4 + 168*c8) / 40320
    
    c12 = -(c0 + 12.0*c4 + 168.0*c8) / 40320.0
    
    # The test function is f(x) = g(x) * exp(-x^2/4) ? Or exp(-x^2/2)?
    # The task says "Hermite-polynomial times Gaussian test function".
    # Standard Fourier uncertainty inequality involves L^2 norm.
    # A(f) = ||f||_2 / ||f||_1 ? No, A(f) is usually related to the uncertainty principle.
    # Let's look at the seed program values: [-113/100, 1/25, 1/3240]
    # c0 = -1.13, c4 = 0.04, c8 = 1/3240 ~ 0.000308
    # c12 = -(-1.13 + 12*0.04 + 168*(1/3240)) / 40320
    # 12*0.04 = 0.48
    # 168/3240 = 0.05185
    # Sum = -1.13 + 0.48 + 0.05185 = -0.59815
    # c12 = 0.59815 / 40320 ~ 1.48e-5
    
    # The objective function is the upper bound on A(f)*A(Fourier(f)).
    # For a function f(x) = P(x) e^{-x^2/4}, the Fourier transform is also a polynomial times Gaussian.
    # The product of the "uncertainty" measures is bounded.
    # Let's assume the standard form:
    # A(f) = sqrt( ||f||_2^2 / ||f||_1^2 ) ? No.
    # Let's assume the task refers to the specific bound derived in the notebook.
    # Since I don't have the notebook, I must infer the objective.
    # However, the task says "score is the upper bound on C4".
    # And "lower is better".
    # The seed returns a specific vector.
    # If I just return the seed vector, I get 1.0.
    # I need to minimize the objective.
    
    # Let's construct the function f(x) = g(x) * exp(-x^2/4)
    # Why exp(-x^2/4)? Because H_n(x) e^{-x^2/4} are eigenfunctions of the Fourier transform?
    # Actually, H_n(x) e^{-x^2/2} are eigenfunctions.
    # Let's assume the Gaussian is exp(-x^2/2).
    # f(x) = g(x) exp(-x^2/2)
    
    # We need to compute A(f) and A(F(f)).
    # What is A(f)?
    # In many uncertainty principles, A(f) = ||f||_2 / ||f||_1 is not it.
    # Maybe A(f) = sqrt( \int x^2 |f|^2 dx / \int |f|^2 dx ) ? (Standard deviation)
    # If so, A(f)*A(F(f)) >= 1/4 (Heisenberg).
    # The task mentions "upper bound on A(f)*A(Fourier(f))".
    # This suggests we are looking for the minimum possible value of this product for this class of functions, or rather, the bound is tight for some function in the class?
    # Or maybe we are minimizing the product itself?
    # "Tune ... to reduce the upper bound". This implies the bound is a function of the coefficients, and we want to minimize it.
    
    # Let's assume A(f) is the standard deviation of |f|^2.
    # sigma_x^2 = \int x^2 |f(x)|^2 dx / \int |f(x)|^2 dx
    # sigma_k^2 = \int k^2 |F(k)|^2 dk / \int |F(k)|^2 dk
    # Product P = sigma_x * sigma_k.
    
    # Let's compute this numerically.
    
    # Define g(x)
    def g(x):
        return (c0 * 1.0 +
                c4 * (16.0*x**4 - 48.0*x**2 + 12.0) +
                c8 * (256.0*x**8 - 1344.0*x**6 + 2688.0*x**4 - 1344.0*x**2 + 168.0) +
                c12 * (4096.0*x**12 - 46080.0*x**10 + 241920.0*x**8 - 552960.0*x**6 + 645120.0*x**4 - 322560.0*x**2 + 40320.0))
    
    # f(x) = g(x) * exp(-x^2/2)
    # We need to integrate over a sufficient range.
    # Since g is a polynomial, the integrals are finite.
    # We can use Gauss-Hermite quadrature?
    # Or just dense sampling.
    
    # Let's use dense sampling for robustness.
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]
    
    f = g(x) * np.exp(-x**2 / 2.0)
    f2 = f**2
    
    norm_sq = np.trapz(f2, x)
    if norm_sq < 1e-10:
        return 1e10 # Avoid division by zero
    
    mean_x2 = np.trapz(x**2 * f2, x) / norm_sq
    sigma_x = np.sqrt(mean_x2)
    
    # Fourier Transform of f(x) = g(x) exp(-x^2/2)
    # F(k) = 1/sqrt(2*pi) * \int f(x) e^{-ikx} dx
    # We can compute the Fourier transform numerically using FFT or direct integration.
    # Direct integration is safer for accuracy with polynomials.
    # Or use the property that FT of H_n(x) exp(-x^2/2) is (-i)^n H_n(k) exp(-k^2/2).
    # This is exact!
    # F(k) = 1/sqrt(2*pi) * [ c0 * H_0(k) + c4 * (-i)^4 H_4(k) + c8 * (-i)^8 H_8(k) + c12 * (-i)^12 H_12(k) ] * exp(-k^2/2)
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
    # So F(k) is proportional to g(k) exp(-k^2/2) as well?
    # Wait, (-i)^n for even n is 1.
    # So F(k) = 1/sqrt(2*pi) * g(k) exp(-k^2/2).
    # This means the function is an eigenfunction of the Fourier transform (with eigenvalue 1/sqrt(2*pi) * something? No, the eigenvalue is (-i)^n for H_n. For a sum, it's a mix. But since all n are even, they all have eigenvalue 1).
    # Actually, FT[H_n(x) e^{-x^2/2}] = (-i)^n H_n(k) e^{-k^2/2}.
    # So if g(x) = sum c_n H_n(x), then F(k) = 1/sqrt(2*pi) sum c_n (-i)^n H_n(k) e^{-k^2/2}.
    # Since n=0,4,8,12 are all even, (-i)^n = 1.
    # So F(k) = 1/sqrt(2*pi) g(k) e^{-k^2/2}.
    
    # Therefore, sigma_k is identical to sigma_x!
    # sigma_k = sigma_x.
    # Product P = sigma_x^2.
    
    # So we just need to minimize sigma_x^2 = \int x^2 g(x)^2 e^{-x^2} dx / \int g(x)^2 e^{-x^2} dx.
    
    # Let's compute this ratio.
    # Numerator: \int x^2 g(x)^2 e^{-x^2} dx
    # Denominator: \int g(x)^2 e^{-x^2} dx
    
    # We can use Gauss-Hermite quadrature for high precision and speed.
    # np.polynomial.hermite.hermgauss(n)
    # The weight is exp(-x^2).
    
    # Let's use n=100 points for Gauss-Hermite.
    x_gh, w_gh = np.polynomial.hermite.hermgauss(100)
    
    g_vals = g(x_gh)
    g2 = g_vals**2
    
    denom = np.sum(w_gh * g2)
    if denom < 1e-10:
        return 1e10
        
    numer = np.sum(w_gh * x_gh**2 * g2)
    
    sigma_x2 = numer / denom
    
    return sigma_x2

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # 180s limit, keep margin
    
    # Initial guess from seed
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Try to improve with L-BFGS-B
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        if time.time() < deadline - 10.0:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    except Exception:
        pass
    
    # Try Nelder-Mead for local refinement if time permits
    if time.time() < deadline - 10.0:
        try:
            res_nm = minimize(_objective, best_x, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass
    
    # Ensure we return the best found
    return best_x

# EVOLVE-BLOCK-END
