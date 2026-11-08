"""Program to minimize the upper bound C4 for the uncertainty inequality using Hermite-Gaussian test functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial

_start_time = time.time()
_TIME_LIMIT = 160.0  # safety margin before 180s kill

def _compute_moments_analytical(a0, a4, a8):
    """
    Compute moments analytically for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    
    Using orthogonality: ∫H_n(x)H_m(x)exp(-x^2)dx = n! * sqrt(pi) * delta_nm
    
    For ∫x^2 g^2 dx, use x*H_n = (1/2)H_{n+1} + n*H_{n-1}
    So x^2*H_n = (1/4)H_{n+2} + n*H_n + (n/4)*H_{n-2}
    """
    # a12 from g(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 10395.0
    
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # ∫g^2 dx = sum_i c_i^2 * i! * sqrt(pi)
    norm_sq = 0.0
    for c, n in zip(coeffs, degrees):
        norm_sq += c**2 * float(factorial(n)) * np.sqrt(np.pi)
    
    if norm_sq < 1e-30:
        return norm_sq, 0.0, 0.0
    
    # ∫x^2 g^2 dx
    # x^2 * H_n = (1/4)H_{n+2} + n*H_n + (n/4)*H_{n-2} (for n>=2)
    # For n=0: x^2*H_0 = x^2 = (1/4)H_2 + 0*H_0 + ... 
    # Actually: x*H_0 = H_1/2, so x^2*H_0 = x*H_1/2 = (H_2/4 + H_0/2)/2 = H_2/4 + H_0/4
    # Let me be more careful:
    # x*H_n = (1/2)H_{n+1} + n*H_{n-1}
    # x^2*H_n = x*((1/2)H_{n+1} + n*H_{n-1}) = (1/2)[(1/2)H_{n+2} + (n+1)H_n] + n[(1/2)H_n + (n-1)H_{n-2}]
    # = (1/4)H_{n+2} + (n+1)/2 * H_n + n/2 * H_n + n(n-1)/2 * H_{n-2}
    # = (1/4)H_{n+2} + (2n+1)/2 * H_n + n(n-1)/2 * H_{n-2}
    
    # So ∫x^2 g^2 dx = sum_{i,j} c_i c_j ∫x^2 H_i H_j exp(-x^2) dx
    # = sum_{i,j} c_i c_j [ (1/4)∫H_{i+2}H_j exp(-x^2) + (2i+1)/2 ∫H_i H_j exp(-x^2) + i(i-1)/2 ∫H_{i-2}H_j exp(-x^2) ]
    # Using orthogonality: ∫H_m H_j exp(-x^2) = m! sqrt(pi) delta_{mj}
    
    moment_sq = 0.0
    for i, (ci, ni) in enumerate(zip(coeffs, degrees)):
        for j, (cj, nj) in enumerate(zip(coeffs, degrees)):
            # Term 1: (1/4) * ∫H_{ni+2} H_{nj} exp(-x^2) = (1/4) * (ni+2)! * sqrt(pi) * delta_{ni+2, nj}
            if ni + 2 == nj:
                term1 = 0.25 * float(factorial(ni + 2)) * np.sqrt(np.pi)
            else:
                term1 = 0.0
            
            # Term 2: (2*ni+1)/2 * ∫H_{ni} H_{nj} exp(-x^2) = (2*ni+1)/2 * ni! * sqrt(pi) * delta_{ni, nj}
            if ni == nj:
                term2 = (2*ni + 1) / 2.0 * float(factorial(ni)) * np.sqrt(np.pi)
            else:
                term2 = 0.0
            
            # Term 3: ni*(ni-1)/2 * ∫H_{ni-2} H_{nj} exp(-x^2) = ni*(ni-1)/2 * (ni-2)! * sqrt(pi) * delta_{ni-2, nj}
            if ni >= 2 and ni - 2 == nj:
                term3 = ni * (ni - 1) / 2.0 * float(factorial(ni - 2)) * np.sqrt(np.pi)
            else:
                term3 = 0.0
            
            moment_sq += ci * cj * (term1 + term2 + term3)
    
    return norm_sq, moment_sq, a12

def _compute_fourier_moments(a0, a4, a8):
    """
    Compute ∫k^2 |ĝ(k)|^2 dk using the fact that for g(x) = P(x)exp(-x^2/2),
    the Fourier transform can be computed analytically.
    
    For the physicist's Hermite functions, the Fourier transform has known properties.
    Alternatively, use the relation: ∫k^2|ĝ|^2 dk = ∫(g'')^2 dx (up to constants)
    
    Actually, with the convention ĝ(k) = ∫g(x)e^{-ikx}dx, we have:
    ∫k^2|ĝ(k)|^2 dk = ∫(d²g/dx²)^2 dx (by Parseval and integration by parts)
    
    But more directly: for g(x) = P(x)exp(-x^2/2), we can compute the Fourier transform.
    
    Let me use: ∫k^2|ĝ|^2 dk = ∫|g''|^2 dx - ∫|g'|^2 dx ... no, that's not right either.
    
    Actually: F[k^2 ĝ] = -g'' (with appropriate constants)
    So ∫k^2|ĝ|^2 dk = ∫|g''|^2 dx (up to 2π factors)
    
    Let me use a simpler approach: compute numerically but efficiently.
    """
    a0, a4, a8 = a0, a4, a8
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 10395.0
    
    # Use the relation: for g(x) = P(x)exp(-x^2/2)
    # g'(x) = P'(x)exp(-x^2/2) - x*P(x)exp(-x^2/2) = [P'(x) - x*P(x)]exp(-x^2/2)
    # g''(x) = [P''(x) - 2x*P'(x) + (x^2-1)*P(x)]exp(-x^2/2)
    
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # We need P', P''
    
    # H_n' = 2n H_{n-1}
    # H_n'' = 2n * 2(n-1) H_{n-2} = 4n(n-1) H_{n-2}
    
    # Let's compute P', P'' as Hermite combinations
    # P = sum c_n H_n
    # P' = sum c_n * 2n * H_{n-1}
    # P'' = sum c_n * 4n(n-1) * H_{n-2}
    
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # P' coefficients: c_n * 2n at degree n-1
    P_prime_coeffs = {}
    for c, n in zip(coeffs, degrees):
        if n >= 1:
            P_prime_coeffs[n-1] = P_prime_coeffs.get(n-1, 0) + c * 2 * n
    
    # P'' coefficients: c_n * 4n(n-1) at degree n-2
    P_double_prime_coeffs = {}
    for c, n in zip(coeffs, degrees):
        if n >= 2:
            P_double_prime_coeffs[n-2] = P_double_prime_coeffs.get(n-2, 0) + c * 4 * n * (n-1)
    
    # g'' = [P'' - 2x*P' + (x^2-1)*P] exp(-x^2/2)
    # We need ∫(g'')^2 dx = ∫[P'' - 2x*P' + (x^2-1)*P]^2 exp(-x^2) dx
    
    # This is getting complex. Let me use a different approach.
    # 
    # Key insight: For the uncertainty product, we need:
    # Δx^2 = ∫x^2|g|^2 / ∫|g|^2
    # Δk^2 = ∫k^2|ĝ|^2 / ∫|ĝ|^2
    #
    # By Plancherel: ∫|ĝ|^2 dk = (1/2π)∫|g|^2 dx (with convention ĝ = ∫g e^{-ikx}dx)
    # So Δk^2 = ∫k^2|ĝ|^2 / ∫|g|^2 * 2π
    #
    # The uncertainty product is Δx * Δk
    #
    # For the "upper bound C4", it might be Δx * Δk itself.
    #
    # Let me compute ∫k^2|ĝ|^2 dk numerically but efficiently.
    
    x = np.linspace(-20, 20, 8001)
    dx = x[1] - x[0]
    gauss = np.exp(-x**2 / 2.0)
    
    from scipy.special import eval_hermite
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gauss
    
    norm_sq_num = np.sum(g**2) * dx
    moment_sq_num = np.sum(x**2 * g**2) * dx
    
    # Compute Fourier transform numerically
    # Use FFT on a fine grid
    N = 2**14  # 16384
    L = 40.0
    x_fft = np.linspace(-L, L, N, endpoint=False)
    dx_fft = x_fft[1] - x_fft[0]
    k_fft = 2*np.pi*np.fft.fftfreq(N, d=dx_fft)
    
    gauss_fft = np.exp(-x_fft**2 / 2.0)
    H0_fft = np.ones_like(x_fft)
    H4_fft = eval_hermite(4, x_fft)
    H8_fft = eval_hermite(8, x_fft)
    H12_fft = eval_hermite(12, x_fft)
    
    g_fft = (a0 * H0_fft + a4 * H4_fft + a8 * H8_fft + a12 * H12_fft) * gauss_fft
    
    G_fft = np.fft.fft(g_fft) * dx_fft
    # G_fft[k] ≈ ∫g(x)e^{-ikx}dx
    
    fourier_norm_sq = np.sum(np.abs(G_fft)**2) * dx_fft  # ≈ ∫|ĝ|^2 dk
    fourier_moment_sq = np.sum(k_fft**2 * np.abs(G_fft)**2) * dx_fft  # ≈ ∫k^2|ĝ|^2 dk
    
    return norm_sq_num, moment_sq_num, fourier_norm_sq, fourier_moment_sq

def _compute_bound(a):
    """Compute the uncertainty product Δx * Δk"""
    a0, a4, a8 = a
    norm_sq, moment_sq, fourier_norm_sq, fourier_moment_sq = _compute_fourier_moments(a0, a4, a8)
    
    if norm_sq < 1e-30 or fourier_norm_sq < 1e-30:
        return 1e10
    
    var_x = moment_sq / norm_sq
    var_k = fourier_moment_sq / fourier_norm_sq
    
    # Handle negative values from numerical error
    var_x = max(var_x, 0)
    var_k = max(var_k, 0)
    
    uncertainty_product = np.sqrt(var_x) * np.sqrt(var_k)
    return uncertainty_product

def _objective(a):
    return _compute_bound(a)

def construct():
    """
    Returns three coefficients [a0, a4, a8] for the Hermite combination.
    a12 is determined by g(0)=0.
    Minimizes the upper bound on C4 (uncertainty product).
    """
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        return np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Seed values
    best_a = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_bound(best_a)
    
    # Try a structured grid search first to get a good starting point
    # The coefficients are likely small, so search in a reasonable range
    
    # First, try the seed and some nearby points
    candidates = [best_a.copy()]
    
    # Add some hand-crafted candidates based on physics intuition
    # For g(0)=0, we need the polynomial to vanish at 0
    # H4(0)=3, H8(0)=105, H12(0)=10395, H0(0)=1
    # So a0 + 3*a4 + 105*a8 + 10395*a12 = 0
    
    # Try symmetric-ish combinations
    test_points = [
        np.array([-1.0, 0.3, -0.01]),
        np.array([-0.5, 0.15, -0.005]),
        np.array([-2.0, 0.6, -0.02]),
        np.array([-1.0, 0.5, -0.03]),
        np.array([-0.1, 0.01, -0.001]),
        np.array([-10.0, 3.0, -0.1]),
        np.array([-100.0, 30.0, -1.0]),
    ]
    
    for tp in test_points:
        val = _compute_bound(tp)
        if val < best_val:
            best_val = val
            best_a = tp.copy()
    
    # Now do local optimization from the best point
    elapsed = time.time() - _start_time
    if elapsed < _TIME_LIMIT - 60:
        try:
            result = minimize(_objective, best_a, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_a = result.x
        except Exception:
            pass
    
    # Try a few more local optimizations from different starts
    elapsed = time.time() - _start_time
    rng = np.random.default_rng(12345)
    for _ in range(5):
        if elapsed > _TIME_LIMIT - 30:
            break
        x0 = best_a + rng.normal(0, 0.1, size=3)
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_a = result.x
        except Exception:
            continue
    
    return best_a
# EVOLVE-BLOCK-END
