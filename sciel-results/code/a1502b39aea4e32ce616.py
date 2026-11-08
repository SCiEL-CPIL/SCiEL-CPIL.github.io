"""Optimized search for Hermite coefficients minimizing the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_coeffs():
    """Return coefficients for H0, H4, H8, H12 in standard (probabilists') form."""
    # Standard Hermite polynomials (probabilists'):
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 75600x^2 + 10395
    return np.array([1.0, 1.0, 1.0, 1.0])  # placeholder, actual computation below


def _eval_hermite(x, coeffs):
    """Evaluate the Hermite combination at x. coeffs = [c0, c4, c8, c12]."""
    # H0
    h0 = 1.0
    # H4 = x^4 - 6x^2 + 3
    h4 = x**4 - 6.0 * x**2 + 3.0
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    h8 = x**8 - 28.0 * x**6 + 210.0 * x**4 - 420.0 * x**2 + 105.0
    # H12 = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 75600x^2 + 10395
    h12 = (x**12 - 60.0 * x**10 + 1260.0 * x**8 - 12600.0 * x**6 
           + 56700.0 * x**4 - 75600.0 * x**2 + 10395.0)
    
    return coeffs[0] * h0 + coeffs[1] * h4 + coeffs[2] * h8 + coeffs[3] * h12


def _compute_bound(c0, c4, c8):
    """Compute the C4 upper bound given c0, c4, c8 with c12 fixed by g(0)=0."""
    # g(0) = c0 + c4*3 + c8*105 + c12*10395 = 0
    # So c12 = -(c0 + 3*c4 + 105*c8) / 10395
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0
    coeffs = np.array([c0, c4, c8, c12])
    
    # We need to compute A(f) * A(Fourier(f)) for the test function
    # f(x) = g(x) * exp(-x^2/2) where g is the Hermite combination
    # The bound involves integrals of |f| and |f_hat|
    
    # Numerical computation of the bound
    # f(x) = g(x) * exp(-x^2/2)
    # f_hat(k) = integral f(x) exp(-ikx) dx (up to normalization)
    
    # For the C4 bound, we need:
    # C4 = sup over f of A(f) * A(f_hat)
    # where A(f) = sup |f| / (some norm)
    
    # Let's compute numerically:
    # A(f) = max_x |f(x)| / (max_x |f(x)|) ... actually A(f) is typically defined as
    # the ratio of sup norm to L1 norm or similar
    
    # From the context, the bound is on C4 = A(f)*A(F)
    # Let's use numerical integration
    
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    
    # f(x) = g(x) * exp(-x^2/2)
    g_vals = _eval_hermite(x, coeffs)
    gauss = np.exp(-x**2 / 2.0)
    f_vals = g_vals * gauss
    
    # A(f) = sup |f| / (integral |f|) -- or some similar ratio
    # Actually, for the uncertainty inequality context:
    # A(f) = ||f||_inf / ||f||_1 or similar
    
    sup_f = np.max(np.abs(f_vals))
    int_f = np.sum(np.abs(f_vals)) * dx
    
    # Fourier transform
    # f_hat(k) = integral f(x) exp(-ikx) dx
    # For real f, we can use FFT approximation
    # But let's compute directly for accuracy
    
    k = np.linspace(-50, 50, 10000)
    dk = k[1] - k[0]
    
    # Compute f_hat numerically
    # f_hat(k) = integral f(x) exp(-ikx) dx
    # = integral f(x) cos(kx) dx - i * integral f(x) sin(kx) dx
    cos_terms = np.exp(-1j * np.outer(k, x))  # This is too expensive
    # Better: use FFT
    
    # Use FFT-based approach
    # f is defined on x grid, compute its Fourier transform
    # Pad to get better frequency resolution
    n_pad = 32768
    f_padded = np.zeros(n_pad, dtype=complex)
    # Place f in the center
    mid = n_pad // 2
    n_f = len(f_vals)
    start = mid - n_f // 2
    f_padded[start:start+n_f] = f_vals
    
    # FFT
    f_hat = np.fft.fftshift(np.fft.fft(f_padded)) * dx * n_f / n_pad
    
    # Actually, let's be more careful with the FFT scaling
    # f_hat(k_j) ≈ dx * sum_n f(x_n) exp(-i k_j x_n)
    # With FFT, we get values at discrete frequencies
    
    sup_fh = np.max(np.abs(f_hat))
    int_fh = np.sum(np.abs(f_hat)) * dk
    
    # The bound C4 involves A(f) * A(f_hat)
    # A(f) is typically ||f||_inf / ||f||_1
    # But the exact definition matters
    
    # From Goncalves et al., the bound is:
    # A(f) * A(f_hat) <= C4
    # where A(f) = sup_x |f(x)| / (∫|f|)
    
    if int_f < 1e-15 or int_fh < 1e-15:
        return 1e10  # invalid
    
    A_f = sup_f / int_f
    A_fh = sup_fh / int_fh
    
    return A_f * A_fh


def construct():
    """Find optimal Hermite coefficients minimizing the C4 upper bound."""
    t_start = time.time()
    deadline = t_start + 180.0  # Safety margin before 210s kill
    
    # Initial guess from prior work
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_bound = _compute_bound(*best_coeffs)
    
    def objective(params):
        return _compute_bound(params[0], params[1], params[2])
    
    # Multi-start BFGS optimization
    # Grid of initial points
    c0_range = np.linspace(-2.0, 0.5, 15)
    c4_range = np.linspace(-0.5, 0.5, 12)
    c8_range = np.linspace(-0.1, 0.1, 8)
    
    n_starts = len(c0_range) * len(c4_range) * len(c8_range)
    n_done = 0
    
    for c0_init in c0_range:
        for c4_init in c4_range:
            for c8_init in c8_range:
                if time.time() > deadline - 30.0:
                    break
                x0 = np.array([c0_init, c4_init, c8_init])
                try:
                    result = minimize(
                        objective, x0, method='BFGS',
                        options={'gtol': 1e-12, 'maxiter': 200}
                    )
                    if result.success:
                        val = result.fun
                        if val < best_bound:
                            best_bound = val
                            best_coeffs = result.x
                except Exception:
                    pass
                n_done += 1
            if time.time() > deadline - 30.0:
                break
        if time.time() > deadline - 30.0:
            break
    
    # Refine with more precise computation
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                objective, best_coeffs, method='BFGS',
                options={'gtol': 1e-14, 'maxiter': 500}
            )
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array([best_coeffs[0], best_coeffs[1], best_coeffs[2]])
# EVOLVE-BLOCK-END
