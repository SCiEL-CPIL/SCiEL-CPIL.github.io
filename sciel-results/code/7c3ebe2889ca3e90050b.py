"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hermite_h(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H0 = np.ones_like(x)
        H1 = 2 * x
        for k in range(2, n + 1):
            H2 = 2 * x * H1 - 2 * (k - 1) * H0
            H0, H1 = H1, H2
        return H1

def _compute_c4_bound(a0, a1, a2):
    """Compute the upper bound on C4 given the three free coefficients."""
    # H12 coefficient fixed so g(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 1032192.0
    
    a3 = -(a0 * H0_0 + a1 * H4_0 + a2 * H8_0) / H12_0
    
    # Evaluate on a fine grid
    x = np.linspace(-8, 8, 20000)
    
    H0 = _hermite_h(0, x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    g = (a0 * H0 + a1 * H4 + a2 * H8 + a3 * H12) * np.exp(-x**2 / 2)
    
    # For even Hermite polynomials, F(H_n(x) e^{-x^2/2}) = (-1)^n H_n(xi) e^{-xi^2/2} * sqrt(2*pi)
    # Since 4, 8, 12 are all even, (-1)^n = 1
    F_g = (a0 * H0 + a1 * H4 + a2 * H8 + a3 * H12) * np.exp(-x**2 / 2) * np.sqrt(2 * np.pi)
    
    # The C4 bound for the uncertainty inequality is:
    # C4 = (sup_x |g(x)|) * (sup_xi |F(g)(xi)|) / (inf_x |g(x)| * inf_xi |F(g)(xi)|)
    # but this is problematic due to zeros.
    
    # Alternative interpretation: C4 is the best constant such that
    # A(f) * A(F(f)) >= C4 for all f, where A(f) = sup |f/g|
    # This equals the operator norm of the Fourier transform on the weighted space.
    
    # A practical approximation: C4 ≈ (||g||_inf / ||g||_2) * (||F(g)||_inf / ||F(g)||_2)
    # This measures the "peakedness" of g and its Fourier transform.
    
    g_inf = np.max(np.abs(g))
    g_2 = np.sqrt(np.trapz(g**2, x))
    F_g_inf = np.max(np.abs(F_g))
    F_g_2 = np.sqrt(np.trapz(F_g**2, x))
    
    if g_2 < 1e-10 or F_g_2 < 1e-10:
        return 1e10
    
    c4 = (g_inf / g_2) * (F_g_inf / F_g_2)
    
    return c4

def construct():
    """Return three coefficients of the Hermite combination (degrees 0, 4, 8)."""
    start_time = time.time()
    deadline = 175.0  # safety margin
    
    # Seed coefficients
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_x = seed.copy()
    best_c4 = _compute_c4_bound(*seed)
    
    # Try local optimization from the seed
    try:
        result = minimize(_compute_c4_bound, seed, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if result.fun < best_c4:
            best_c4 = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    # Try from nearby points
    if time.time() - start_time < deadline - 20.0:
        for delta in [0.01, 0.05, 0.1]:
            for sign in [-1, 1]:
                x_try = seed + sign * delta * np.array([1, 1, 1])
                try:
                    c4_try = _compute_c4_bound(*x_try)
                    if c4_try < best_c4:
                        best_c4 = c4_try
                        best_x = x_try.copy()
                except Exception:
                    pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END

