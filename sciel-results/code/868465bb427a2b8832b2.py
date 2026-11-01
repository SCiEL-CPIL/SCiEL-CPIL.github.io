"""
Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
the upper bound on A(f)*A(Fourier(f)).

Uses analytic L2 norms and fast numerical L1 norms for efficient optimization.
"""

import numpy as np
from scipy import optimize
import time

# Precompute Hermite polynomials on a grid for L1 norms
# Use a grid that is large enough to capture the tails of the Gaussian
_NUM_POINTS = 10000
_X_GRID = np.linspace(-15, 15, _NUM_POINTS)
_DX = _X_GRID[1] - _X_GRID[0]
_GAUSS = np.exp(-_X_GRID**2 / 2.0)

# Precompute Hermite polynomials H0, H4, H8, H12
_H0 = np.ones_like(_X_GRID)
_H4 = np.zeros_like(_X_GRID)
_H8 = np.zeros_like(_X_GRID)
_H12 = np.zeros_like(_X_GRID)

# Compute Hermite polynomials using recurrence or numpy
# H0 = 1
# H1 = 2x
# Hn+1 = 2x Hn - 2n Hn-1
def _compute_hermite_grid():
    H = [np.ones_like(_X_GRID), 2 * _X_GRID]
    for n in range(1, 13):
        Hn_plus_1 = 2 * _X_GRID * H[n] - 2 * n * H[n-1]
        H.append(Hn_plus_1)
    return H

_H_list = _compute_hermite_grid()
_H0 = _H_list[0]
_H4 = _H_list[4]
_H8 = _H_list[8]
_H12 = _H_list[12]

# Precompute weights for L2 norm: sqrt(pi) * 2^n * n!
# L2^2 = sqrt(pi) * sum c_n^2 * 2^n * n!
import math
_L2_WEIGHTS = {}
for n in [0, 4, 8, 12]:
    _L2_WEIGHTS[n] = math.sqrt(math.pi) * (2 ** n) * math.factorial(n)

def _compute_c12(c0, c4, c8):
    """Compute c12 from the constraint g(0) = 0."""
    # H0(0) = 1, H4(0) = 12, H8(0) = 1190, H12(0) = 113880
    return -(c0 + 12.0 * c4 + 1190.0 * c8) / 113880.0

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on C4 given coefficients c0, c4, c8.
    """
    c12 = _compute_c12(c0, c4, c8)
    
    # L2 norm squared (analytic)
    l2_sq = (_L2_WEIGHTS[0] * c0**2 + 
             _L2_WEIGHTS[4] * c4**2 + 
             _L2_WEIGHTS[8] * c8**2 + 
             _L2_WEIGHTS[12] * c12**2)
    
    if l2_sq < 1e-12:
        return 1e10
        
    l2_norm = np.sqrt(l2_sq)
    
    # L1 norm of g (numeric)
    g = (c0 * _H0 + c4 * _H4 + c8 * _H8 + c12 * _H12) * _GAUSS
    l1_g = np.trapz(np.abs(g), _X_GRID)
    
    # L1 norm of Fourier transform (numeric)
    # F(g)(k) = integral g(x) exp(-2*pi*i*x*k) dx
    # We can compute this on the same grid for k
    # Note: The grid for k should be appropriate for the frequency content
    # Since g is a polynomial times Gaussian, its FT is also a polynomial times Gaussian
    # The FT of H_n(x) exp(-x^2/2) is (-i)^n H_n(k) exp(-k^2/2) (with scaling)
    # Let's use the property that the FT of the Hermite function is itself (up to phase)
    # psi_n(x) = H_n(x) exp(-x^2/2)
    # F(psi_n)(k) = (-i)^n psi_n(k) (in the unitary convention)
    # In the non-unitary convention F(f)(k) = int f(x) e^{-2*pi*i*x*k} dx:
    # F(psi_n)(k) = (-i*2*pi)^n * sqrt(1/(2*pi)) * H_n(k) * exp(-k^2/2) ? 
    # Actually, it's easier to just compute the FT numerically using FFT or direct integration
    # But direct integration is slow.
    # Let's use the fact that the FT of a Hermite Gaussian is another Hermite Gaussian.
    # Specifically, if g(x) = sum c_n H_n(x) exp(-x^2/2), then
    # F(g)(k) = sum c_n (-i)^n H_n(k) exp(-k^2/2) * (some constant)
    # The constant depends on the normalization.
    # Let's verify with n=0: g(x) = exp(-x^2/2). F(g)(k) = sqrt(2*pi) exp(-2*pi^2 k^2).
    # This does NOT look like H_0(k) exp(-k^2/2).
    # So the simple eigenfunction property holds for the unitary FT with frequency variable omega.
    # F_omega(f)(omega) = (1/sqrt(2*pi)) int f(x) e^{-i x omega} dx.
    # Then F_omega(psi_n)(omega) = (-i)^n psi_n(omega).
    # The task likely uses the standard math convention or the one in the paper.
    # Given the ambiguity, and that the score is an "upper bound", it's safer to compute
    # the L1 norm of the FT numerically using FFT on a sufficiently large grid.
    
    # Let's use FFT for the FT L1 norm
    # We need to be careful with the scaling and frequency axis
    N = len(_X_GRID)
    if N % 2 != 0:
        # Make it even
        N -= 1
        x = _X_GRID[:N]
        g = g[:N]
        dx = _DX
    else:
        x = _X_GRID
        g = g
        dx = _DX
        
    # FFT
    G_fft = np.fft.fft(g) * dx
    G_shifted = np.fft.fftshift(G_fft)
    
    # Frequency axis
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    l1_ft = np.trapz(np.abs(G_shifted), k)
    
    # Product
    bound = (l1_g / l2_norm) * (l1_ft / l2_norm)
    
    return bound

def construct():
    """
    Find optimal coefficients c0, c4, c8 to minimize the upper bound.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = start_time + 200.0
    
    # Starting point
    x0 = np.array([-1.13, 0.04, 0.0003])
    
    def objective(coeffs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        try:
            c0, c4, c8 = coeffs
            c0 = np.clip(c0, -100, 100)
            c4 = np.clip(c4, -100, 100)
            c8 = np.clip(c8, -100, 100)
            return _compute_bound(c0, c4, c8)
        except Exception:
            return 1e10
    
    best_coeffs = x0.copy()
    best_val = objective(x0)
    
    # Use L-BFGS-B with numerical gradients (fast enough with analytic L2)
    try:
        result = optimize.minimize(objective, x0, method='L-BFGS-B', 
                                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < best_val:
            best_val = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    # Polish with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            result = optimize.minimize(objective, best_coeffs, method='Nelder-Mead',
                                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
            
    return np.array(best_coeffs)

if __name__ == "__main__":
    coeffs = construct()
    print(f"Optimal coefficients: {coeffs}")
    bound = _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    print(f"Upper bound on C4: {bound}")
