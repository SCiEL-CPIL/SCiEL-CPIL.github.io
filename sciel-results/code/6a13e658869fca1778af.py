"""
Optimized program for the third autocorrelation inequality.
Combines Chebyshev polynomial initialization with direct time-domain 
soft-max optimization.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective_soft_direct(h, beta):
    """
    Soft-max surrogate objective in time domain.
    h is normalized internally so sum(h) = N to maintain scale invariance 
    for the optimization landscape, but the exact value is computed 
    on the normalized version to match the evaluator's normalization 
    implicitly if we assume sum(h)=N is the optimal scale.
    Actually, the value is scale invariant: V(c*h) = V(h).
    So we can just normalize h to sum N for numerical stability in the objective.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _chebyshev_basis(k, x):
    """
    Compute Chebyshev polynomials of the second kind U_0..U_{k-1} at points x.
    U_0(x) = 1
    U_1(x) = 2x
    U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
    """
    U = np.zeros((k, len(x)))
    U[0, :] = 1.0
    if k > 1:
        U[1, :] = 2.0 * x
    for i in range(2, k):
        U[i, :] = 2.0 * x * U[i-1, :] - U[i-2, :]
    return U

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses Chebyshev polynomial initialization followed by direct 
    time-domain optimization with a soft-max surrogate.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Start with all-ones (value 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Chebyshev Polynomial Initialization
    # We parameterize h as a linear combination of the first M Chebyshev polynomials.
    # h[i] = sum_{k=0}^{M-1} c_k U_k(x_i)
    # We want to minimize the autocorrelation peak.
    # The grid x_i is on [-1, 1].
    x_grid = np.linspace(-1.0, 1.0, N)
    
    # Try a few different numbers of basis functions and random coefficients
    rng = np.random.RandomState(42)
    
    for M in [5, 10, 20]:
        if time.time() > deadline - 100:
            break
        U = _chebyshev_basis(M, x_grid)
        
        # Try random coefficient vectors
        for trial in range(50):
            if time.time() > deadline - 100:
                break
            coeffs = rng.randn(M) * 0.5
            h = U.T @ coeffs
            # Ensure sum is not zero and finite
            s = np.sum(h)
            if abs(s) < 1e-6 or not np.all(np.isfinite(h)):
                continue
            # Normalize to sum N for consistent scale during evaluation
            h_norm = h * (N / s)
            v = _eval_value(h_norm)
            if v < best_val:
                best_val = v
                best_h = h_norm.copy()

    # Also try the simple cosine family as a fallback if Chebyshev didn't improve much
    if best_val > 1.5:
        for m in range(1, 21):
            if time.time() > deadline - 90:
                break
            for a in np.linspace(-0.8, 0.8, 17):
                h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # Phase 2: Direct Time-Domain Optimization with Soft-Max
    # We optimize the raw vector h directly.
    # We use a multi-stage approach with increasing beta for the soft-max.
    
    # Initial point
    h_current = best_h.copy()
    
    # Beta schedule: start low for smooth landscape, increase for precision
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        def obj_func(h_vec, beta=beta):
            return _objective_soft_direct(h_vec, beta)
        
        # We perform a few random restarts around the current best
        # but for high beta, we stick closer to the previous best to avoid getting stuck in local minima of the sharper landscape.
        restarts = 5 if beta < 50 else 2
        
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
                
            if r == 0:
                h_start = h_current.copy()
            else:
                # Add small noise scaled by the previous beta's "temperature"
                noise_scale = 0.1 / (1 + r)
                h_start = h_current + noise_scale * rng.randn(N)
            
            try:
                result = minimize(
                    obj_func,
                    h_start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    # Evaluate exact value
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        h_current = h_opt.copy()
            except Exception:
                continue

    # Phase 3: Final Polish with very high beta
    if time.time() < deadline - 5:
        beta_final = 5000.0
        
        def obj_final(h_vec):
            return _objective_soft_direct(h_vec, beta_final)
        
        try:
            result = minimize(
                obj_final,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = result.x
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _eval_value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
