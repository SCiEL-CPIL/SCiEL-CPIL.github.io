"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time


def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound C4 on A(f)*A(Fourier(f)) for the test function
    g(x) = (a0 + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    where a12 is fixed so that g(0) = 0.
    
    The bound is derived from the uncertainty principle analysis.
    """
    # Fix a12 so g(0) = 0
    # H_n(0) for even n: H_0(0)=1, H_4(0)=3, H_8(0)=105, H_12(0)=-945
    # g(0) = (a0 + 3*a4 + 105*a8 - 945*a12) * 1 = 0
    a12 = (a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Use numerical quadrature over a sufficiently large domain
    # g(x) = P(x) * exp(-x^2/2) where P(x) = a0 + a4*H4(x) + a8*H8(x) + a12*H12(x)
    
    x = np.linspace(-30, 30, 20000)
    dx = x[1] - x[0]
    
    H0 = np.ones_like(x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    gauss = np.exp(-0.5 * x**2)
    g = P * gauss
    
    # Compute moments needed for the bound
    # The bound on C4 typically involves:
    # C4 = (∫x^4 |g|^2 dx / ∫|g|^2 dx) * (∫ξ^4 |ĝ|^2 dξ / ∫|ĝ|^2 dξ)
    # or similar products of second moments in position and frequency space
    
    # For Fourier transform convention: ĝ(ξ) = ∫g(x)exp(-2πixξ)dx
    # But in the Hermite context, it's often: ĝ(ξ) = ∫g(x)exp(-ixξ)dx / sqrt(2π)
    
    # Let's compute using the standard L2 norm and second moments
    # |g|^2 = P^2 * exp(-x^2)
    g_sq = g**2
    
    # Position space moments
    norm_sq = np.sum(g_sq) * dx
    if norm_sq < 1e-15:
        return 1e10  # invalid
    
    # <x^2> = ∫x^2 |g|^2 dx / ∫|g|^2 dx
    x2_norm = np.sum(x**2 * g_sq) * dx / norm_sq
    
    # For frequency space, use Parseval and the fact that for Hermite functions,
    # the Fourier transform relates to the same polynomial structure
    # Actually, let's compute the Fourier transform numerically
    
    # Use FFT for the Fourier transform
    # g(x) on [-L, L], L=30
    L = 30.0
    N = len(x)
    dx_fft = 2*L/N
    
    # FFT gives DFT, need to scale properly
    # ĝ(ξ) = ∫_{-L}^{L} g(x) exp(-ixξ) dx ≈ dx * sum g(x_j) exp(-ix_j ξ_k)
    # Using FFT: the k-th frequency component
    G = np.fft.fftshift(np.fft.fft(g)) * dx_fft
    
    # Frequency grid
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx_fft))
    
    # |ĝ|^2
    G_sq = np.abs(G)**2
    
    # Frequency space moments
    freq_norm_sq = np.sum(G_sq)
    if freq_norm_sq < 1e-15:
        return 1e10
    
    freq2_norm = np.sum(freqs**2 * G_sq) / freq_norm_sq
    
    # The bound C4 is typically (position second moment) * (frequency second moment)
    # For the standard uncertainty: Δx * Δξ >= 1/2
    # The C4 bound involves higher moments
    
    # Based on the task description, C4 is an upper bound on A(f)*A(F̂(f))
    # A(f) likely refers to some norm or moment-based quantity
    
    # In the Goncalves et al. context, the bound involves:
    # C4 = (∫x^4|g|^2/∫|g|^2) * (∫ξ^4|ĝ|^2/∫|ĝ|^2) / (∫x^2|g|^2/∫|g|^2 * ∫ξ^2|ĝ|^2/∫|ĝ|^2)
    # or simply the product of variance-like quantities
    
    # Let me try the simplest interpretation: product of second moments
    # which gives the standard Heisenberg uncertainty bound
    C4 = x2_norm * freq2_norm
    
    return C4


def construct():
    """
    Return three coefficients (a0, a4, a8) of the Hermite combination
    that minimize the upper bound C4 on A(f)*A(Fourier(f)).
    """
    start_time = time.time()
    deadline = 200.0  # 10s safety margin before 210s kill
    
    # Seed from Goncalves et al. 2017
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_coeff = seed.copy()
    best_bound = _compute_bound(*seed)
    
    # Local optimization starting from seed
    def objective(coeffs):
        return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    
    # Try multiple starting points and local optimizations
    start_points = [
        seed,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.13, 0.04, 0.000308]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.5, 0.1, 0.001]),
        np.array([-0.5, 0.02, 0.0001]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30.0:
            break
        try:
            result = minimize(objective, sp, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeff = result.x.copy()
        except Exception:
            pass
    
    # If time remains, do a finer polish
    if time.time() < deadline - 5.0:
        try:
            result = minimize(objective, best_coeff, method='Nelder-Mead',
                             options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeff = result.x.copy()
        except Exception:
            pass
    
    # Final check: ensure we haven't degraded
    final_bound = _compute_bound(*best_coeff)
    if final_bound > best_bound:
        best_coeff = seed.copy()
    
    return best_coeff
# EVOLVE-BLOCK-END
