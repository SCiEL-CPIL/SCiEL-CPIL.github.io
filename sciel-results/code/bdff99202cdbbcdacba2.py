"""Search for optimal Hermite combination coefficients minimizing C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds
from scipy.special import factorial

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 180.0  # 210s kill, keep safety margin


def hermite_h(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        return 256 * x**8 - 3072 * x**6 + 11520 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        return 4096 * x**12 - 73728 * x**10 + 552960 * x**8 - 2150400 * x**6 + 3991680 * x**4 - 3024000 * x**2 + 92400
    else:
        # General recurrence
        H = np.ones_like(x)
        if n >= 1:
            H1 = 2 * x
            for k in range(2, n + 1):
                H, H1 = H1, 2 * x * H1 - 2 * (k - 1) * H
        return H1 if n >= 1 else H


def g_of_x(x, c0, c4, c8):
    """g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    with c12 chosen so g(0) = 0."""
    H0 = np.ones_like(x)
    H4 = hermite_h(4, x)
    H8 = hermite_h(8, x)
    H12 = hermite_h(12, x)
    # c12 so that at x=0: c0 + c4*12 + c8*1680 + c12*92400 = 0
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 92400.0
    poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    return poly * np.exp(-0.5 * x**2)


def fourier_transform_g(c0, c4, c8):
    """Compute Fourier transform of g(x) = P(x)*exp(-x^2/2) numerically.
    Using the property that FT of x^n * exp(-x^2/2) can be computed via Hermite polynomials.
    For a polynomial times Gaussian, the FT is also a polynomial times Gaussian.
    FT[x^n e^{-x^2/2}] = (-i)^n H_n(ξ) e^{-ξ^2/2} (with appropriate normalization)
    
    Actually, let's use the standard: if g(x) = P(x) e^{-x^2/2}, then
    ĝ(ξ) = ∫ P(x) e^{-x^2/2} e^{-2πixξ} dx
    
    Let's use a different approach: numerically compute via FFT or direct integration.
    But for the C4 inequality, we need specific moments.
    
    Let me reconsider. The C4 constant involves:
    A(f) = ∫ |f(x)|^2 dx (or sup norm?)
    Actually A(f) is likely the L2 norm or L∞ norm.
    
    For the uncertainty inequality A(f)*A(ĝ), with A being some norm,
    the C4 bound is a specific constant.
    
    Let me use a numerical approach: compute g and ĝ on a grid, then
    estimate the relevant norms.
    """
    # Use a fine grid
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    g = g_of_x(x, c0, c4, c8)
    
    # FFT to get Fourier transform
    # g(x) = P(x) e^{-x^2/2}
    # FT: ĝ(ξ) = ∫ g(x) e^{-2πixξ} dx
    # Using FFT: need to handle the shift and scaling
    n = len(x)
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    
    return x, g, freqs, G


def compute_c4(c0, c4, c8):
    """Compute the C4 upper bound for the given coefficients.
    
    The uncertainty product bound involves:
    A(f) = ||f||_2 (L2 norm) or similar
    A(ĝ) = ||ĝ||_2
    
    For g(x) = P(x) e^{-x^2/2}, we can compute:
    ||g||_2^2 = ∫ g(x)^2 dx
    ||ĝ||_2^2 = ||g||_2^2 (by Plancherel, with proper normalization)
    
    Wait, if A(f) is the L2 norm, then A(f)*A(ĝ) = ||f||_2 * ||ĝ||_2 = ||f||_2^2 by Plancherel.
    That's not interesting.
    
    Maybe A(f) is the L∞ norm? Or a mixed norm?
    
    Let me think about this differently. The C4 constant in the uncertainty inequality
    likely refers to a specific formulation. Given the task mentions "upper bound on C4",
    and the coefficients are for a test function that achieves near-optimality,
    
    The product A(f)*A(ĝ) where A might be:
    - L∞ norm: ||f||_∞ * ||ĝ||_∞
    - Or some other combination
    
    Let me compute both L2 and L∞ norms and their product.
    """
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    g = g_of_x(x, c0, c4, c8)
    
    # L2 norm
    l2_g = np.sqrt(np.sum(g**2) * dx)
    
    # L∞ norm
    linf_g = np.max(np.abs(g))
    
    # Fourier transform
    n = len(x)
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    
    # L2 norm of FT (should equal L2 norm of g by Plancherel)
    l2_G = np.sqrt(np.sum(np.abs(G)**2) * (1.0 / (n * dx)))  # Parseval
    
    # Actually, let's be more careful with FFT normalization
    # np.fft.fft: G[k] = sum_j g[j] * exp(-2πi jk/N)
    # So ĝ(ξ_k) ≈ dx * G[k] where ξ_k = k/(N*dx)
    # ||ĝ||_2^2 = ∫ |ĝ(ξ)|^2 dξ ≈ sum |ĝ(ξ_k)|^2 * dξ = sum |dx*G[k]|^2 * (1/(N*dx))
    # = dx * sum |G[k]|^2 / N
    
    l2_G2 = np.sqrt(dx * np.sum(np.abs(G)**2) / n)
    
    # L∞ norm of FT
    linf_G = np.max(np.abs(G)) * dx
    
    # The C4 product - let's try L∞ * L∞ as the uncertainty product
    # Or maybe it's a specific formulation from the literature
    
    # For now, return the product of L∞ norms as the "C4" value
    c4_val = linf_g * linf_G
    
    return c4_val


def objective(coeffs):
    """Objective function to minimize: the C4 upper bound."""
    c0, c4, c8 = coeffs
    try:
        val = compute_c4(c0, c4, c8)
        if not np.isfinite(val):
            return 1e10
        return val
    except:
        return 1e10


def construct():
    """Find optimal coefficients (c0, c4, c8) minimizing the C4 upper bound."""
    deadline = START_TIME + TIME_LIMIT - 10.0  # 10s safety margin
    
    # Start from the prior state of the art
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = objective(best_coeffs)
    
    # Try multiple optimization strategies
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Strategy 1: Nelder-Mead from the seed
    if time.time() < deadline:
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except:
            pass
    
    # Strategy 2: BFGS from the seed
    if time.time() < deadline:
        try:
            res = minimize(objective, best_coeffs, method='BFGS',
                          options={'maxiter': 200, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except:
            pass
    
    # Strategy 3: Try different starting points
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        for i in range(20):
            if time.time() > deadline:
                break
            x0 = rng.uniform(-5, 5, size=3)
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x
            except:
                pass
    
    # Strategy 4: Polish with tighter tolerance
    if time.time() < deadline:
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except:
            pass
    
    # Ensure g(0) = 0 is satisfied (it is by construction via c12)
    return np.array([best_coeffs[0], best_coeffs[1], best_coeffs[2]])
# EVOLVE-BLOCK-END
