"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

_START_TIME = time.time()
_TIME_LIMIT = 200.0  # safety margin before 210s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _hermite_n(x, n):
    """Physicists' Hermite polynomials via recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 = A(f) * A(F(f)) for the test function
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0.
    
    The uncertainty bound involves the L1 norms of |g| and |G| (Fourier transform).
    For Fourier-invariant analysis, we compute:
      A(g) = integral |g(x)| dx
      A(G) = integral |G(x)| dx (Fourier transform)
    and the bound C4 = A(g) * A(G).
    
    We use the standard Fourier transform convention:
      G(x) = (1/sqrt(2*pi)) * integral g(t) exp(-i x t) dt
    which is unitary, so Plancherel holds: ||g||_2 = ||G||_2.
    
    The Hermite functions h_n(x) = H_n(x) * exp(-x^2/2) / sqrt(2^n n! sqrt(pi))
    are eigenfunctions of the Fourier transform with eigenvalues (-i)^n.
    
    For our g(x) = sum_k c_k H_k(x) exp(-x^2/2),
    the normalized Hermite function basis is phi_k(x) = H_k(x) exp(-x^2/2) / sqrt(2^k k! sqrt(pi)).
    
    So g(x) = sum_k c_k sqrt(2^k k! sqrt(pi)) * phi_k(x)
    
    The Fourier transform: F[phi_k] = (-i)^k phi_k
    
    So G(x) = sum_k c_k sqrt(2^k k! sqrt(pi)) * (-i)^k phi_k(x)
    
    We need to compute A(g) = integral |g(x)| dx and A(G) = integral |G(x)| dx.
    """
    # Choose a12 so that g(0) = 0
    # g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0)
    # H0(0) = 1, H4(0) = -12, H8(0) = 1344, H12(0) = -1032192
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1344.0
    H12_0 = -1032192.0
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # Compute g(x) on a grid
    x = np.linspace(-8, 8, 20000)
    dx = x[1] - x[0]
    
    g = np.zeros_like(x)
    for c, d in zip(coeffs, degrees):
        g += c * _hermite_n(x, d) * np.exp(-x**2 / 2)
    
    # Compute A(g) = integral |g(x)| dx
    A_g = np.trapz(np.abs(g), x)
    
    # Compute G(x) using the Fourier transform property of Hermite functions
    # G(x) = sum_k c_k sqrt(2^k k! sqrt(pi)) * (-i)^k * phi_k(x)
    # where phi_k(x) = H_k(x) exp(-x^2/2) / sqrt(2^k k! sqrt(pi))
    # So G(x) = sum_k c_k (-i)^k H_k(x) exp(-x^2/2) / sqrt(2^k k! sqrt(pi)) * sqrt(2^k k! sqrt(pi))
    # Wait, let me redo this more carefully.
    
    # g(x) = sum_k c_k H_k(x) e^{-x^2/2}
    # phi_k(x) = H_k(x) e^{-x^2/2} / N_k where N_k = sqrt(2^k k! sqrt(pi))
    # So H_k(x) e^{-x^2/2} = N_k * phi_k(x)
    # g(x) = sum_k c_k N_k phi_k(x)
    # F[g](x) = sum_k c_k N_k (-i)^k phi_k(x)
    
    G = np.zeros_like(x)
    for c, d in zip(coeffs, degrees):
        N_d = np.sqrt(2**d * float(np.math.factorial(d)) * np.sqrt(np.pi))
        # (-i)^k: for k=0: 1, k=4: 1, k=8: 1, k=12: 1 (all multiples of 4)
        sign = 1.0 if (d % 4 == 0) else (-1.0 if (d % 4 == 2) else (1j if d % 4 == 1 else -1j))
        # For degrees 0,4,8,12, all are divisible by 4, so (-i)^d = 1
        G += c * N_d * sign * _hermite_n(x, d) * np.exp(-x**2 / 2) / N_d
        # Simplifies to: G += c * sign * H_d(x) e^{-x^2/2}
    
    # Since all degrees are multiples of 4, (-i)^d = 1 for all, so G = g
    # This makes sense: the function is invariant under Fourier transform (up to sign)
    
    A_G = np.trapz(np.abs(G), x)
    
    return A_g * A_G

def construct():
    # Start from the known prior state of the art
    x0 = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    best_x = x0.copy()
    best_val = _compute_bound(*x0)
    
    # Try multiple optimization methods and restarts
    methods = ['Nelder-Mead', 'Powell', 'BFGS']
    
    for method in methods:
        if _time_left() < 10:
            break
        
        try:
            result = minimize(
                lambda v: _compute_bound(*v),
                x0,
                method=method,
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Try perturbations around the best point found
    if _time_left() > 20:
        for scale in [0.01, 0.001, 0.1]:
            if _time_left() < 5:
                break
            for i in range(3):
                if _time_left() < 2:
                    break
                perturbation = np.random.normal(0, scale, size=3)
                candidate = best_x + perturbation
                val = _compute_bound(*candidate)
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
    
    # Final polish with tight tolerances
    if _time_left() > 15:
        try:
            result = minimize(
                lambda v: _compute_bound(*v),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
