"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_phys(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    H0 = np.ones_like(x)
    H1 = 2.0 * x
    for k in range(2, n + 1):
        H2 = 2.0 * x * H1 - 2.0 * (k - 1) * H0
        H0, H1 = H1, H2
    return H1


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 for a test function of the form
    f(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    where a12 is determined by the constraint f(0) = 0.
    
    The objective is the uncertainty product Delta_x * Delta_k for this
    Fourier-invariant function. Since the function is a Fourier eigenfunction
    (all degrees are multiples of 4, so (-i)^n = 1), Delta_x = Delta_k,
    and the product is Delta_x^2.
    
    We minimize this to get the tightest uncertainty bound.
    """
    # Compute a12 from constraint f(0) = 0
    a12 = -(a0 + 12*a4 + 1680*a8) / 2032128.0
    
    # Use numerical integration on a grid
    x = np.linspace(-12, 12, 2000)
    dx = x[1] - x[0]
    
    H0 = np.ones_like(x)
    H4 = _hermite_phys(4, x)
    H8 = _hermite_phys(8, x)
    H12 = _hermite_phys(12, x)
    
    f = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * np.exp(-x**2/2)
    
    # L2 norm squared
    norm_sq = np.sum(f**2) * dx
    if norm_sq < 1e-30:
        return 1e10
    
    # Second moment <x^2>
    x2_norm = np.sum(x**2 * f**2) * dx
    var_x = x2_norm / norm_sq  # Since <x> = 0 by symmetry
    
    # For Fourier eigenfunctions with (-i)^n = 1, Delta_k = Delta_x
    # So the uncertainty product is var_x (which is Delta_x^2)
    bound = var_x
    
    return bound


def construct():
    start_time = time.time()
    deadline = 180.0  # seconds
    
    # Initial guess from the seed
    x0_seed = np.array([-1.13, 0.04, 0.00030864])
    
    # Multi-start with diverse initial guesses
    np.random.seed(42)
    starts = [x0_seed]
    
    # Add diverse random starts
    for i in range(5):
        starts.append(np.array([
            np.random.uniform(-3, 3),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.01, 0.01)
        ]))
    
    best_x = x0_seed.copy()
    best_val = _compute_bound(*x0_seed)
    
    for x0 in starts:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        try:
            result = minimize(_compute_bound, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            val = result.fun
            if val < best_val:
                best_val = val
                best_x = result.x.copy()
                
        except Exception:
            continue
    
    return best_x

# EVOLVE-BLOCK-END
