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

def _objective_norm(h, beta=100.0, lam_sum=1000.0):
    """
    Objective for normalized optimization.
    We want to minimize 2*N*max(conv) subject to sum(h)=1.
    Since scale is invariant, we can fix sum(h)=1.
    Penalty for sum(h) != 1.
    """
    s = np.sum(h)
    # Penalty for sum not being 1
    penalty = lam_sum * (s - 1.0)**2
    if abs(s) < 1e-10:
        return 1e10 + penalty
    conv = _fft_autoconv(h)
    soft_max, _ = _soft_max_and_weights(conv, beta)
    # Note: if we enforce sum=1, the value is 2*N*soft_max
    return 2.0 * N * soft_max + penalty

def _objective_norm_grad(h, beta=100.0, lam_sum=1000.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    soft_max, weights = _soft_max_and_weights(conv, beta)
    
    if np.max(weights) < 1e-15:
        grad_sm = np.zeros_like(h)
    else:
        grad_sm = _grad_autoconv(weights, h)
        
    grad_penalty = 2.0 * lam_sum * (s - 1.0) * np.ones_like(h)
    
    return 2.0 * N * grad_sm + grad_penalty

def _refine(h, deadline):
    """Refine a candidate h using a sequence of optimization phases."""
    h_curr = h.copy()
    
    # Normalize initial guess to sum=1 to start in the right manifold
    s0 = np.sum(h_curr)
    if abs(s0) > 1e-6:
        h_curr = h_curr / s0
    else:
        h_curr = np.ones(N) / N

    remaining = deadline - time.time()
    
    if remaining < 5.0:
        return h_curr

    # Phase 1: Low beta, strong sum penalty
    if remaining > 15.0:
        try:
            res = minimize(_objective_norm, h_curr, method='L-BFGS-B',
                           args=(20.0, 1000.0), jac=_objective_norm_grad,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
            h_curr = res.x
        except Exception:
            pass
            
    # Phase 2: Medium beta
    if deadline - time.time() > 10.0:
        try:
            res = minimize(_objective_norm, h_curr, method='L-BFGS-B',
                           args=(50.0, 1000.0), jac=_objective_norm_grad,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7})
            h_curr = res.x
        except Exception:
            pass

    # Phase 3: High beta, weaker sum penalty (sum should be close to 1)
    if deadline - time.time() > 5.0:
        try:
            res = minimize(_objective_norm, h_curr, method='L-BFGS-B',
                           args=(100.0, 100.0), jac=_objective_norm_grad,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
            h_curr = res.x
        except Exception:
            pass

    # Phase 4: Very high beta polish
    if deadline - time.time() > 2.0:
        try:
            res = minimize(_objective_norm, h_curr, method='L-BFGS-B',
                           args=(200.0, 10.0), jac=_objective_norm_grad,
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-11})
            h_curr = res.x
        except Exception:
            pass
            
    # Final normalization
    s_final = np.sum(h_curr)
    if abs(s_final) > 1e-6:
        h_curr = h_curr / s_final
        
    return h_curr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(12345)
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    i = np.arange(N)
    initializations = []
    
    # 1. Seed
    initializations.append(np.ones(N))
    
    # 2. Low-frequency sinusoids
    for k in [1, 2, 3, 4, 5]:
        for a in [0.5, 1.0, 2.0]:
            h0 = a + np.sin(2*np.pi*k*i/N)
            initializations.append(h0)
            
    # 3. Cosine variants
    for k in [1, 2, 3]:
        for a in [1.0, 2.0]:
            h0 = a + np.cos(2*np.pi*k*i/N)
            initializations.append(h0)
            
    # 4. Comb patterns
    for period in [2, 3, 4, 5, 6, 8, 10]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 += 0.5 * rng.standard_normal(N)
        if abs(np.sum(h0)) > 1.0:
            initializations.append(h0)
            
    # 5. Low-pass filtered noise
    for _ in range(10):
        h0 = rng.standard_normal(N)
        kernel = np.ones(9) / 9.0
        h0 = np.convolve(h0, kernel, mode='same')
        h0 += 1.0
        initializations.append(h0)
        
    # 6. Random smooth functions (Fourier series)
    for _ in range(10):
        coeffs = rng.standard_normal(10)
        h0 = np.zeros(N)
        for j, c in enumerate(coeffs):
            h0 += c * np.sin(2*np.pi*(j+1)*i/N)
        h0 += 2.0 + np.abs(h0).mean() # Ensure positive mean
        
    initializations.append(h0)
    for _ in range(10):
        coeffs = rng.standard_normal(10)
        h0 = np.zeros(N)
        for j, c in enumerate(coeffs):
            h0 += c * np.sin(2*np.pi*(j+1)*i/N)
        h0 += 2.0 + np.abs(h0).mean()
        initializations.append(h0)

    # 7. Unitary transformations of sparse impulses
    for _ in range(10):
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
        initializations.append(h_rot)

    # Optimize each initialization
    candidates = []
    
    for idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
            
        h_refined = _refine(h0, deadline)
        val = _evaluate(h_refined)
        if val < best_val:
            best_val = val
            best_h = h_refined.copy()
            
        candidates.append((val, h_refined))
        
    # Sort candidates
    candidates.sort(key=lambda x: x[0])
    
    # Take top 5 and run a final, very short high-precision polish
    top_candidates = candidates[:5]
    
    for val, h_cand in top_candidates:
        if time.time() > deadline - 5.0:
            break
        # Quick polish
        try:
            res = minimize(_objective_norm, h_cand, method='L-BFGS-B',
                           args=(300.0, 1.0), jac=_objective_norm_grad,
                           options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-12})
            h_polished = res.x
            s_final = np.sum(h_polished)
            if abs(s_final) > 1e-6:
                h_polished = h_polished / s_final
            val_polished = _evaluate(h_polished)
            if val_polished < best_val:
                best_val = val_polished
                best_h = h_polished
        except Exception:
            pass

    return best_h
