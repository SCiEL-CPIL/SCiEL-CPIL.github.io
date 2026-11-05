"""Optimized program for third autocorrelation inequality using Chebyshev polynomial parameterization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 135.0  # safety margin before 150s kill


def _autoconv_fft(h):
    """Compute full autoconvolution using FFT."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * n - 1]


def _objective(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _chebyshev_basis(M, x):
    """
    Compute the first M Chebyshev polynomials of the second kind U_k(x) evaluated at points x.
    Returns array of shape (M, len(x)) where row k is U_k(x).
    """
    n = len(x)
    basis = np.zeros((M, n))
    if M > 0:
        basis[0, :] = 1.0
    if M > 1:
        basis[1, :] = 2.0 * x
    for k in range(2, M):
        basis[k, :] = 2.0 * x * basis[k - 1, :] - basis[k - 2, :]
    return basis


def _construct_from_coeffs(coeffs, M):
    """
    Given coefficients for Chebyshev polynomials of the second kind, construct h.
    h[j] = sum_k coeffs[k] * U_k(x_j) + offset
    where x_j are uniform grid points in [-1, 1].
    We include an offset (constant term) as part of the parameterization.
    """
    # Uniform grid on [-1, 1]
    x = np.linspace(-1, 1, N, endpoint=False)
    # Actually, let's use a grid that covers [-1/4, 1/4] mapped to [-1, 1]
    # The step function is on [-1/4, 1/4] with N intervals.
    # Let's just use x in [-1, 1] for the Chebyshev evaluation.
    
    if M <= 0:
        return np.ones(N)
    
    basis = _chebyshev_basis(M, x)
    # coeffs has length M, where coeffs[0] is the constant term
    h = np.zeros(N)
    for k in range(M):
        h += coeffs[k] * basis[k, :]
    
    return h


def construct():
    start_time = time.time()
    
    # Start with seed
    best_h = np.ones(N)
    best_val = _objective(best_h)  # should be 2.0
    
    rng = np.random.RandomState(12345)
    
    # Grid points for Chebyshev evaluation
    x = np.linspace(-1, 1, N, endpoint=False)
    
    # Try different numbers of Chebyshev modes
    M_values = [2, 3, 4, 5, 6, 8, 10, 12]
    
    for M in M_values:
        if time.time() - start_time > DEADLINE:
            break
        
        # Precompute basis for this M
        basis = _chebyshev_basis(M, x)
        
        # Define objective in terms of coefficients
        def obj_coeffs(coeffs):
            h = np.zeros(N)
            for k in range(len(coeffs)):
                h += coeffs[k] * basis[k, :]
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10
            conv = _autoconv_fft(h)
            return abs(2.0 * N * np.max(conv) / (s * s))
        
        # Try multiple random initializations
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Random initialization
            if r == 0:
                # Start near constant: large constant term, small others
                coeffs_init = np.zeros(M)
                coeffs_init[0] = 1.0
                for k in range(1, M):
                    coeffs_init[k] = rng.randn() * 0.01
            else:
                coeffs_init = rng.randn(M) * 0.1
                coeffs_init[0] += 1.0  # Ensure non-zero sum
            
            # Optimize
            try:
                res = minimize(obj_coeffs, coeffs_init, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_val:
                    h_new = np.zeros(N)
                    for k in range(M):
                        h_new += res.x[k] * basis[k, :]
                    val_new = _objective(h_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_h = h_new.copy()
            except Exception:
                pass
    
    # Additional refinement: try to optimize the full vector h directly
    # starting from the best Chebyshev-based solution
    if time.time() - start_time < DEADLINE - 10:
        def obj_full(h):
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10
            conv = _autoconv_fft(h)
            return abs(2.0 * N * np.max(conv) / (s * s))
        
        try:
            res = minimize(obj_full, best_h.copy(), method='L-BFGS-B',
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                val_new = _objective(res.x)
                if val_new < best_val:
                    best_val = val_new
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Final verification
    final_val = _objective(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h


# Quick test
if __name__ == '__main__':
    h = construct()
    val = _objective(h)
    print(f"Value: {val}")
    print(f"Score: {2.0 / val}")
