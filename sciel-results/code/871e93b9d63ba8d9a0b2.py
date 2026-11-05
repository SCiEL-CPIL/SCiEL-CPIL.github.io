import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct
import time

N = 400

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n
    fh = np.fft.rfft(h, size)
    conv = np.fft.irfft(fh * fh, size)[:n + n - 1]
    return conv

def _evaluate(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv)) / (s * s)

def _soft_max_and_weights(conv, beta):
    """Compute soft-max and its weights for the autoconvolution."""
    m = np.max(conv)
    if m <= 0:
        return m, np.zeros_like(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = m + np.log(sum_exp) / beta
    return soft_max, weights

def _grad_autoconv(w, h):
    """
    Gradient of sum_j w_j * conv_j w.r.t. h.
    """
    L = 2 * N
    w_pad = np.zeros(L)
    w_pad[:len(w)] = w
    h_pad = np.zeros(L)
    h_pad[:N] = h
    
    W = np.fft.fft(w_pad, n=L)
    H = np.fft.fft(h_pad, n=L)
    
    cross = np.fft.ifft(W * np.conj(H)).real
    grad_h = 2.0 * cross[:N]
    return grad_h

def _objective_soft(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_autoconv(h)
    soft_max, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _objective_soft_grad(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    soft_max, weights = _soft_max_and_weights(conv, beta)
    
    if np.max(weights) < 1e-15:
        return np.zeros_like(h)
    
    grad_sm = _grad_autoconv(weights, h)
    
    return 2.0 * N * (grad_sm / (s * s) - 2.0 * soft_max * np.ones_like(h) / (s * s * s))

def _spectral_penalty(h, lam=0.1):
    """Penalty for spectral concentration (L4 norm of DCT coefficients)."""
    dct_coeffs = dct(h, type=2, norm='ortho')
    return lam * np.sum(np.abs(dct_coeffs)**4) / N

def _spectral_penalty_grad(h, lam=0.1):
    """Gradient of spectral penalty."""
    dct_coeffs = dct(h, type=2, norm='ortho')
    grad = 4.0 * lam / N * idct(dct_coeffs**3, type=2, norm='ortho')
    return grad

def _combined_obj(h, beta=50.0, lam=0.05):
    return _objective_soft(h, beta) + _spectral_penalty(h, lam)

def _combined_grad(h, beta=50.0, lam=0.05):
    return _objective_soft_grad(h, beta) + _spectral_penalty_grad(h, lam)

def _unitary_initializations(rng, count=15):
    """
    Generate initializations using random unitary transformations of sparse impulses.
    This enforces a flat power spectrum (via Parseval) which helps minimize peak-to-average ratio.
    """
    inits = []
    for _ in range(count):
        h_sparse = np.zeros(N)
        num_impulses = rng.integers(5, 20)
        idxs = rng.choice(N, size=num_impulses, replace=False)
        h_sparse[idxs] = rng.choice([-1.0, 1.0], size=num_impulses)
        
        H = np.fft.rfft(h_sparse)
        phases = np.exp(1j * rng.uniform(0, 2*np.pi, size=len(H)))
        H_rot = H * phases
        h_rot = np.fft.irfft(H_rot, n=N)
        
        mean = np.mean(h_rot)
        h_rot = h_rot - mean + 1.0
        
        inits.append(h_rot)
    return inits

def _blashke_product_initializations(rng, count=10):
    """
    Generate initializations based on the real part of Blaschke products.
    The magnitude of the derivative of a Blaschke product has controlled peak-to-mean properties.
    """
    inits = []
    i = np.arange(N)
    z = i / N - 0.5  # Map to [-0.5, 0.5]
    
    for _ in range(count):
        # Place random zeros in the unit disk (mapped to our domain)
        num_zeros = rng.integers(2, 6)
        zeros = []
        for _ in range(num_zeros):
            r = rng.uniform(0.1, 0.9)
            theta = rng.uniform(0, 2*np.pi)
            zeros.append(r * np.exp(1j * theta))
        
        # Compute Blaschke product B(z)
        # For our purposes, we use the real part of the log-derivative or similar
        # Actually, let's use the magnitude of the Blaschke product itself
        # |B(z)| = prod | (z - a_k) / (1 - conj(a_k) z) |
        # But z is real in [-0.5, 0.5]. Let's map to unit disk first.
        # Let w = 2*z, so w in [-1, 1].
        w = 2 * z
        
        log_mag = np.zeros(N)
        for a in zeros:
            # | (w - a) / (1 - conj(a) w) |
            num = np.abs(w - a)
            den = np.abs(1 - np.conj(a) * w)
            # Avoid log(0)
            log_mag += np.log(np.maximum(num, 1e-10) / np.maximum(den, 1e-10))
        
        # h is proportional to exp(log_mag) or 1/exp(log_mag)
        # We want to minimize peak of autoconvolution.
        # Let's try h = 1 / (1 + exp(log_mag)) or similar smooth functions.
        # Or simply h = exp(-alpha * log_mag)
        alpha = rng.uniform(0.5, 2.0)
        h = np.exp(-alpha * log_mag)
        
        # Normalize mean to be positive
        h = h - np.min(h) + 0.5
        inits.append(h)
    return inits

def _refine(h, deadline):
    """Refine a candidate h using a sequence of optimization phases."""
    h_curr = h.copy()
    remaining = deadline - time.time()
    
    if remaining < 3.0:
        return h_curr
    
    # Phase 1: Moderate beta, spectral flattening
    if remaining > 12.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(30.0, 0.1), jac=_combined_grad,
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 2: Higher beta, less spectral penalty
    if deadline - time.time() > 8.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(60.0, 0.02), jac=_combined_grad,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 3: Very high beta, no spectral penalty - peak polishing
    if deadline - time.time() > 4.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(100.0,), jac=_objective_soft_grad,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 4: Extremely high beta for final polish
    if deadline - time.time() > 2.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(200.0,), jac=_objective_soft_grad,
                           options={'maxiter': 100, 'ftol': 1e-13, 'gtol': 1e-11})
            h_curr = res.x
        except Exception:
            pass
            
    return h_curr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s before 150s kill)
    
    rng = np.random.default_rng(12345)
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Generate diverse initializations
    initializations = []
    i = np.arange(N)
    
    # 1. Seed
    initializations.append(np.ones(N))
    
    # 2. Low-frequency sinusoids with DC offset
    for k in [1, 2, 3, 4, 5, 6]:
        for a in [0.5, 1.0, 2.0, 3.0]:
            h0 = a + np.sin(2*np.pi*k*i/N)
            initializations.append(h0)
    
    # 3. Cosine variants
    for k in [1, 2, 3]:
        for a in [1.0, 2.0]:
            h0 = a + np.cos(2*np.pi*k*i/N)
            initializations.append(h0)
            
    # 4. Comb patterns with small noise
    for period in [2, 3, 4, 5, 6, 8, 10, 12]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) > 1.0:
            initializations.append(h0)
            
    # 5. Low-pass filtered noise with DC
    for _ in range(8):
        h0 = rng.standard_normal(N)
        kernel = np.ones(7) / 7.0
        h0 = np.convolve(h0, kernel, mode='same')
        h0 += 1.5
        initializations.append(h0)
        
    # 6. Random smooth functions
    for _ in range(8):
        f = rng.uniform(0.1, 3.0)
        phase = rng.uniform(0, 2*np.pi)
        h0 = np.sin(2*np.pi*f*i/N + phase)
        h0 += 1.0 + 0.5*rng.uniform()
        initializations.append(h0)
    
    # 7. Bump-like functions (Gaussian bumps)
    for _ in range(5):
        center = rng.uniform(0, N)
        width = rng.uniform(20, 80)
        h0 = np.exp(-0.5*((i - center)/width)**2)
        h0 += 0.5
        initializations.append(h0)

    # 8. Unitary transformations of sparse impulses (from inspiration)
    initializations.extend(_unitary_initializations(rng, count=15))
    
    # 9. Blaschke product based initializations (from strategy)
    initializations.extend(_blashke_product_initializations(rng, count=10))

    # Optimize each initialization
    for idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
            
        h_refined = _refine(h0, deadline)
        val = _evaluate(h_refined)
        if val < best_val:
            best_val = val
            best_h = h_refined.copy()
            
    # Final polish on the best found
    if time.time() < deadline - 2.0:
        h_final = _refine(best_h, deadline)
        val_final = _evaluate(h_final)
        if val_final < best_val:
            best_h = h_final
            
    return best_h
