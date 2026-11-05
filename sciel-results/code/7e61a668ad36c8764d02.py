"""Program using sparse DCT representation and Differential Evolution for autocorrelation inequality."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct, idct

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 145.0

def _check_time():
    return time.time() < _deadline - 5.0

def _autoconv(h):
    """Full autoconvolution via FFT: O(N log N)."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _compute_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    max_val = np.max(conv)
    raw = 2.0 * N * max_val / (s * s)
    return abs(raw)

def _dct_to_h(coeffs):
    """Convert DCT coefficients to time-domain heights.
    coeffs[0] is the DC component (mean * sqrt(N)).
    coeffs[k] for k>0 are the amplitudes of the cosine modes.
    """
    # scipy.fft.idct expects type=2 by default? No, default is type=2 for dct, type=3 for idct?
    # Actually, scipy.fft.dct(x, type=2) is the standard DCT-II.
    # The inverse is idct(x, type=3) scaled by 1/(2N) or similar depending on normalization.
    # Let's stick to a simple linear basis to avoid normalization confusion.
    # Basis: phi_k(t) = cos(pi * k * t / N)
    # h[n] = sum_{k=0}^{K-1} c_k * cos(pi * k * n / N)
    # This is a valid basis for symmetric-ish functions, but DCT-II is orthogonal.
    # Let's use the orthogonal DCT-II basis explicitly.
    
    # DCT-II definition: X_k = 2 * sum_{n=0}^{N-1} x_n * cos(pi * (2n+1) * k / (2N))
    # Inverse: x_n = (1/N) * sum_{k=0}^{N-1}'' X_k * cos(pi * (2n+1) * k / (2N))
    # where '' means k=0 term is halved.
    
    # Let's just build the matrix for the first K modes.
    n = np.arange(N)
    k = np.arange(len(coeffs))
    # Argument for cos: pi * (2n + 1) * k / (2N)
    arg = np.pi * np.outer(n, k) / N + np.pi * k / (2 * N)
    basis = np.cos(arg)
    
    # Apply normalization factors for orthogonality
    # For k=0, factor is 1/sqrt(N) in forward, so inverse is sqrt(N)?
    # Actually, let's just use the unnormalized basis vectors and let the optimizer figure out the scale.
    # The basis vectors are:
    # v_k[n] = cos(pi * (2n+1) * k / (2N))
    # These are not normalized to unit length.
    # Length of v_k is sqrt(N/2) for k>0 and sqrt(N) for k=0.
    # We will normalize them to unit length to make the parameter space well-scaled.
    
    norms = np.sqrt(np.sum(basis**2, axis=0))
    norms[norms == 0] = 1.0
    basis_normed = basis / norms
    
    h = coeffs @ basis_normed.T
    return h

def _h_to_dct(h, K):
    """Project h onto the first K normalized DCT-II basis vectors."""
    n = np.arange(N)
    k = np.arange(K)
    arg = np.pi * np.outer(n, k) / N + np.pi * k / (2 * N)
    basis = np.cos(arg)
    norms = np.sqrt(np.sum(basis**2, axis=0))
    norms[norms == 0] = 1.0
    basis_normed = basis / norms
    coeffs = basis_normed.T @ h
    return coeffs

def _objective_dct(coeffs):
    """Objective function for DCT coefficients."""
    h = _dct_to_h(coeffs)
    return _compute_value(h)

def _grad_objective_dct(coeffs):
    """Gradient of the objective with respect to DCT coefficients.
    Since the mapping is linear, grad_coeff = basis_normed.T @ grad_h.
    We need grad_h from the spatial domain objective.
    """
    h = _dct_to_h(coeffs)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(coeffs)
    
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    raw = 2.0 * N * max_val / (s * s)
    val = abs(raw)
    
    # Gradient of max_val w.r.t h
    dh_conv = np.zeros(N)
    valid_j = np.arange(N)
    km = idx - valid_j
    mask = (km >= 0) & (km < N)
    dh_conv[mask] = 2.0 * h[km[mask]]
    
    # Gradient of raw = 2*N*max_val / s^2
    # d(raw)/d(h) = 2*N * (dh_conv * s - 2*max_val * dh_s) / s^3
    # dh_s = 1 for all i
    grad_h = 2.0 * N * (dh_conv * s - 2.0 * max_val) / (s ** 3)
    
    if raw < 0:
        grad_h = -grad_h
        
    # Project gradient to DCT space
    n = np.arange(N)
    k = np.arange(len(coeffs))
    arg = np.pi * np.outer(n, k) / N + np.pi * k / (2 * N)
    basis = np.cos(arg)
    norms = np.sqrt(np.sum(basis**2, axis=0))
    norms[norms == 0] = 1.0
    basis_normed = basis / norms
    
    grad_coeffs = basis_normed.T @ grad_h
    
    return val, grad_coeffs

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Dimension of the DCT subspace
    # Start with a moderate K. Too small might miss the optimum, too large makes DE slow.
    K = 60
    
    # Define bounds for DCT coefficients
    # The scale of h matters. sum(h) should be non-zero.
    # If we use unit-normalized basis, coefficients are roughly the projection amplitudes.
    # For h=1, DC coeff is sqrt(N) ~ 20. Other coeffs are 0.
    # Let's allow coefficients to range from -20 to 20 initially.
    bounds = [(-20.0, 20.0) for _ in range(K)]
    
    # Phase 1: Differential Evolution in DCT space
    # DE is good for global search in low dimensions.
    if _check_time():
        try:
            res = differential_evolution(
                _objective_dct,
                bounds,
                seed=42,
                maxiter=50, # Keep iter low, popsize will be ~50*10=500 evals per gen? No, popsize is 15*K default?
                popsize=10,  # 10 * K = 600 individuals. 50 gens = 30k evals.
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False, # We will polish separately
                callback=None
            )
            
            if res.success or res.fun < best_val:
                coeffs_best = res.x
                h_candidate = _dct_to_h(coeffs_best)
                val_candidate = _compute_value(h_candidate)
                
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Phase 2: Local Optimization in DCT space (L-BFGS-B)
    # Start from the best DE result or a few diverse seeds
    candidates = []
    if _check_time():
        # Re-use DE result if available, otherwise generate seeds
        if 'res' in locals() and res.x is not None:
            candidates.append(res.x)
        
        # Add some heuristic seeds in DCT space
        # 1. All ones -> DC only
        c_dc = np.zeros(K)
        c_dc[0] = np.sqrt(N) # Approximation of DC component for ones
        candidates.append(c_dc)
        
        # 2. Ramp -> High frequency content
        h_ramp = np.linspace(-1, 1, N)
        candidates.append(_h_to_dct(h_ramp, K))
        
        # 3. Cosine bump
        x = np.linspace(-1, 1, N)
        h_bump = np.exp(-20 * x**2)
        candidates.append(_h_to_dct(h_bump, K))
        
        # 4. Random low-pass
        rng = np.random.default_rng(123)
        h_rand = rng.standard_normal(N)
        # Low pass filter in DCT domain
        c_rand = _h_to_dct(h_rand, K)
        c_rand[K//2:] = 0
        candidates.append(c_rand)
        
        for c_init in candidates:
            if not _check_time():
                break
            
            try:
                res_local = minimize(
                    _grad_objective_dct,
                    c_init,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_loc = _dct_to_h(res_local.x)
                val_loc = _compute_value(h_loc)
                
                if val_loc < best_val:
                    best_val = val_loc
                    best_h = h_loc
            except Exception:
                continue

    # Phase 3: Polish in full space (400D)
    # The DCT optimization might get stuck in a local min of the reduced space.
    # A final polish in full space can exploit high-frequency components if they help.
    if best_val < 2.0 and _check_time():
        try:
            # Use the current best_h as start
            res_full = minimize(
                lambda h: (_compute_value(h), _grad_spatial(h)), # Need spatial grad
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            h_full = res_full.x
            val_full = _compute_value(h_full)
            
            if val_full < best_val:
                best_val = val_full
                best_h = h_full
        except Exception:
            pass

    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = _autoconv(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h

def _grad_spatial(h):
    """Spatial gradient for final polish."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)
    
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    raw = 2.0 * N * max_val / (s * s)
    
    dh_conv = np.zeros(N)
    valid_j = np.arange(N)
    km = idx - valid_j
    mask = (km >= 0) & (km < N)
    dh_conv[mask] = 2.0 * h[km[mask]]
    
    grad = 2.0 * N * (dh_conv * s - 2.0 * max_val) / (s ** 3)
    
    if raw < 0:
        grad = -grad
        
    return grad
# EVOLVE-BLOCK-END
