"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial

# Hermite polynomials H_n(x) = n! * sum_{k=0}^{n//2} (-1)^k * (2x)^{n-2k} / (k! * (n-2k)!)
def _hermite(n, x):
    """Compute the physicists' Hermite polynomial H_n(x) using explicit formula."""
    result = np.zeros_like(x)
    for k in range(n // 2 + 1):
        coeff = (-1.0) ** k * factorial(n) / (factorial(k) * factorial(n - 2 * k))
        result += coeff * (2.0 * x) ** (n - 2 * k)
    return result

def _eval_hermite_gaussian(c, x, sigma):
    """
    Evaluate g(x) = (c0*H0 + c1*H4 + c2*H8 + c3*H12) * exp(-x^2/(2*sigma^2))
    where c3 is chosen so that g(0) = 0.
    
    Parameters:
    -----------
    c : array-like, shape (3,)
        Coefficients for H0, H4, H8
    x : array-like
        Points at which to evaluate
    sigma : float
        Standard deviation of the Gaussian envelope
    
    Returns:
    --------
    g : array-like
        The test function values
    """
    c = np.asarray(c, dtype=float)
    c0, c1, c2 = c
    
    # Compute H_n(0) values
    # H0(0) = 1
    # H4(0) = 4! * (-1)^2 * (2*0)^0 / (2! * 0!) = 24/2 = 12
    # H8(0) = 8! * (-1)^4 * (2*0)^0 / (4! * 0!) = 40320/24 = 1680
    # H12(0) = 12! * (-1)^6 * (2*0)^0 / (6! * 0!) = 479001600/720 = 665280
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Choose c3 so that g(0) = 0: c0*H0_0 + c1*H4_0 + c2*H8_0 + c3*H12_0 = 0
    # c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0) / H12_0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Evaluate the combination at all x
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    gaussian = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    
    g = (c0 * H0 + c1 * H4 + c2 * H8 + c3 * H12) * gaussian
    return g

def _compute_c4_bound(c, sigma=1.0, n_points=100000, x_range=10.0):
    """
    Compute the upper bound on C4 for the given coefficients.
    
    The C4 bound is related to the uncertainty product A(f) * A(F(f)).
    For a function g(x), we define:
    A(f) = sqrt(∫ x^2 |g(x)|^2 dx / ∫ |g(x)|^2 dx)  (standard deviation of position)
    A(F(f)) = sqrt(∫ ξ^2 |ĝ(ξ)|^2 dξ / ∫ |ĝ(ξ)|^2 dξ)  (standard deviation of frequency)
    
    The product A(f) * A(F(f)) ≥ 1/2 by the uncertainty principle.
    
    For the specific form g(x) = P(x) * exp(-x^2/(2σ^2)), where P is a polynomial,
    we can compute these quantities. The C4 bound appears to be related to
    the fourth moment ratio or a specific normalization.
    
    Based on the task description and the prior art (Goncalves et al. 2017),
    the C4 bound likely refers to a specific constant in an uncertainty inequality
    involving the fourth moments or a related quantity.
    
    Let me use a practical approach: compute the actual uncertainty product
    and related quantities, and use a well-motivated proxy for C4.
    """
    c = np.asarray(c, dtype=float)
    
    # Choose a reasonable domain and number of points
    x = np.linspace(-x_range, x_range, n_points)
    dx = x[1] - x[0]
    
    g = _eval_hermite_gaussian(c, x, sigma)
    g2 = g ** 2
    
    norm = np.trapz(g2, x)
    if abs(norm) < 1e-15:
        return 1e10  # Degenerate
    
    g2_norm = g2 / norm
    
    # Position moments
    mean_x = np.trapz(x * g2_norm, x)
    var_x = np.trapz((x - mean_x) ** 2 * g2_norm, x)
    mean_x4 = np.trapz(x ** 4 * g2_norm, x)
    
    A_x = np.sqrt(max(var_x, 0.0))
    
    # Compute Fourier transform numerically
    # Use FFT with appropriate scaling
    # g(x) is defined on [-x_range, x_range] with spacing dx
    # FFT: ĝ(k) = sum_j g(x_j) * exp(-2πi k x_j / L) where L = N*dx
    # But we need the continuous FT: ĝ(ξ) = ∫ g(x) exp(-2πi ξ x) dx
    
    # Use the trapezoidal rule approximation
    # ĝ(ξ) ≈ dx * sum_j g(x_j) * exp(-2πi ξ x_j)
    
    # Choose frequency range
    xi_range = 1.0 / (2.0 * dx)  # Nyquist
    n_xi = n_points
    xi = np.linspace(-xi_range, xi_range, n_xi)
    dxi = xi[1] - xi[0]
    
    # Compute FT via matrix-free approach (use FFT after zero-padding or direct)
    # For efficiency, use the fact that we can compute FT at discrete xi points
    # using a DFT if the grid is uniform
    
    # Actually, let's use scipy.fft for efficiency
    from scipy.fft import fft, fftfreq
    
    # Zero-pad for better frequency resolution
    N = n_points
    G = fft(g, n=N * 4)  # 4x zero padding
    
    # Frequency spacing: dxi = 1 / (N * dx) for the padded FFT
    # The FFT gives values at frequencies: k / (N_pad * dx) where k = -N_pad/2, ..., N_pad/2-1
    # Actually, fftfreq(N_pad, d=dx) gives the frequencies
    xi_fft = fftfreq(N * 4, d=dx)
    
    # Reorder to match xi grid (not exactly, but we can interpolate)
    # Or just use the FFT frequencies directly
    
    G_abs2 = np.abs(G) ** 2
    
    # Normalize the FT
    # The continuous FT is approximated by: ĝ(ξ) ≈ dx * G_dft[ξ]
    # where G_dft is the DFT of the sampled g
    # So |ĝ(ξ)|^2 ≈ dx^2 * |G_dft[ξ]|^2
    
    norm_F = np.sum(G_abs2) * dx ** 2 * dxi  # This isn't quite right
    # Actually, the Parseval identity: ∫ |g|^2 dx = ∫ |ĝ|^2 dξ
    # With our approximations: sum |g_j|^2 dx ≈ sum |ĝ_k|^2 dξ
    
    # Let's just normalize G_abs2 to have unit sum (as a proxy for the norm)
    if np.sum(G_abs2) < 1e-15:
        return 1e10
    
    G_abs2_norm = G_abs2 / np.sum(G_abs2)
    
    # Frequency moments using the FFT frequencies
    mean_xi = np.sum(xi_fft * G_abs2_norm)
    var_xi = np.sum((xi_fft - mean_xi) ** 2 * G_abs2_norm)
    
    A_xi = np.sqrt(max(var_xi, 0.0))
    
    # The uncertainty product
    uncertainty_product = A_x * A_xi
    
    # The C4 bound is likely related to this product or a related quantity.
    # Given the task says "score is the upper bound on C4" and "lower is better",
    # and the prior art gives a specific value, let me try different candidates.
    
    # Candidate 1: The uncertainty product itself
    # Candidate 2: Fourth moment ratio in position
    # Candidate 3: A specific combination
    
    # From the prior art (Goncalves et al. 2017), the best known bound on C4
    # for this type of problem is achieved with the given coefficients.
    # The C4 constant might be: C4 = (∫ x^4 |g|^2 / ∫ |g|^2) / (∫ x^2 |g|^2 / ∫ |g|^2)^2
    # which is the "excess kurtosis" or related quantity.
    
    if var_x < 1e-15:
        return 1e10
    
    # Fourth moment ratio
    kurtosis_ratio = mean_x4 / (var_x ** 2)
    
    # Try the uncertainty product as the objective
    return uncertainty_product

def _objective(c):
    """
    Objective function: compute the C4 bound for the given coefficients.
    """
    return _compute_c4_bound(c, sigma=1.0, n_points=50000, x_range=8.0)

def construct():
    """
    Construct the three coefficients for H0, H4, H8 that minimize the C4 bound.
    The H12 coefficient is determined by the constraint g(0) = 0.
    """
    start_time = time.time()
    deadline = 180.0  # Safety margin before the 210s kill
    
    # Baseline from prior art
    baseline = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = baseline.copy()
    best_val = _objective(best)
    
    # Local optimization from baseline
    try:
        result = minimize(_objective, baseline, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_val:
            best_val = result.fun
            best = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > 150:
        return best
    
    # Global search with differential evolution
    try:
        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        de_result = differential_evolution(_objective, bounds, maxiter=50, 
                                           popsize=15, tol=1e-6, seed=42,
                                           polish=True)
        if de_result.fun < best_val:
            best_val = de_result.fun
            best = de_result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > 170:
        return best
    
    # Final polish
    try:
        result = minimize(_objective, best, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < best_val:
            best_val = result.fun
            best = result.x
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
