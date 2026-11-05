"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fh = np.fft.rfft(h, size)
    conv = np.fft.irfft(fh * fh, size)[:2 * n - 1]
    return conv

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_and_weights(conv, beta):
    """Compute soft-max and its weights for the autoconvolution."""
    m = np.max(conv)
    if m <= 0:
        return m, np.zeros_like(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp < 1e-300:
        return m, np.zeros_like(conv)
    weights = exp_shifted / sum_exp
    soft_max = m + np.log(sum_exp) / beta
    return soft_max, weights

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
    
    # Gradient of soft_max w.r.t h
    # soft_max = log(sum(exp(beta*(conv - m)))) + m
    # d(soft_max)/dh = beta * sum(weights * d(conv)/dh)
    # d(conv)/dh is the cross-correlation of weights and h (times 2 for symmetry in autoconv)
    # Actually, let's use the explicit formula for autoconv gradient.
    # conv_k = sum_i h_i h_{i+k}
    # d(conv_k)/dh_l = h_{l+k} + h_{l-k} (with appropriate bounds)
    # This is equivalent to 2 * cross_correlation(weights, h)
    
    L = 2 * N
    w_pad = np.zeros(L)
    w_pad[:len(weights)] = weights
    h_pad = np.zeros(L)
    h_pad[:N] = h
    
    W = np.fft.fft(w_pad, n=L)
    H = np.fft.fft(h_pad, n=L)
    
    # Cross-correlation: sum_j w[j] * h[j-l]
    cross = np.fft.ifft(W * np.conj(H)).real
    grad_sm = 2.0 * cross[:N]
    
    # f = 2*N*soft_max/s^2
    # df/dh_i = 2*N * (grad_sm_i * s^2 - soft_max * 2*s) / s^4
    #         = 2*N * (grad_sm_i * s - 2*soft_max) / s^3
    return 2.0 * N * (grad_sm * s - 2.0 * soft_max) / (s * s * s)

def _spectral_to_h(mags, phases):
    """
    Reconstruct h from spectral magnitudes and phases.
    mags: [m_1, ..., m_{N/2}]
    phases: [p_1, ..., p_{N/2}]
    H_0 is fixed to 1.
    """
    H = np.zeros(N//2 + 1, dtype=complex)
    H[0] = 1.0  # DC component fixed
    
    # Fill positive frequencies
    H[1:] = mags * np.exp(1j * phases)
    
    # Inverse FFT
    h = np.fft.irfft(H, n=N)
    return h

def _spectral_objective(params, beta=50.0):
    mags = params[:N//2]
    phases = params[N//2:]
    h = _spectral_to_h(mags, phases)
    return _objective_soft(h, beta)

def _spectral_grad(params, beta=50.0):
    mags = params[:N//2]
    phases = params[N//2:]
    h = _spectral_to_h(mags, phases)
    
    # Get gradient in spatial domain
    grad_h = _objective_soft_grad(h, beta)
    
    # Chain rule: dObj/dmags, dObj/dphases
    # h = ifft(H)
    # H[0] = 1
    # H[k] = m_k * e^{i p_k}
    
    # d h / d m_k = ifft(d H / d m_k)
    # d h / d p_k = ifft(d H / d p_k)
    
    # We can compute gradients using FFT/IFFT properties or finite differences for simplicity/safety in gradient check,
    # but analytic is better.
    # h_j = (1/N) * (H_0 + 2 * sum_{k=1}^{N/2} m_k cos(2 pi j k / N - p_k))
    # 
    # d h_j / d m_k = (2/N) cos(2 pi j k / N - p_k)
    # d h_j / d p_k = (2/N) m_k sin(2 pi j k / N - p_k)
    
    # grad_mags_k = sum_j grad_h_j * d h_j / d m_k
    # grad_phases_k = sum_j grad_h_j * d h_j / d p_k
    
    j = np.arange(N)
    k = np.arange(1, N//2 + 1)
    
    # Matrix of cos and sin terms
    # shape (N, N/2)
    angles = 2 * np.pi * np.outer(j, k) / N - phases[None, :]
    
    cos_terms = np.cos(angles)
    sin_terms = np.sin(angles)
    
    # grad_h is shape (N,)
    # grad_mags_k = (2/N) * sum_j grad_h_j cos_terms[j, k]
    grad_mags = (2.0 / N) * (cos_terms.T @ grad_h)
    
    # grad_phases_k = (2/N) * sum_j grad_h_j m_k sin_terms[j, k]
    # Note: m_k is shape (N/2,), needs to be broadcasted
    grad_phases = (2.0 / N) * (sin_terms.T @ (grad_h[:, None] * mags[None, :]))
    grad_phases = np.sum(grad_phases, axis=1) # Wait, the dot product handles the sum over j.
    # Let's re-verify the shape.
    # cos_terms.T @ grad_h -> shape (N/2,)
    # sin_terms.T @ (grad_h[:, None] * mags[None, :]) -> shape (N/2, N/2)? No.
    # grad_h[:, None] is (N, 1). mags[None, :] is (1, N/2). Product is (N, N/2).
    # sin_terms is (N, N/2).
    # Element-wise product of sin_terms and (grad_h*mags) is (N, N/2).
    # Sum over j (axis 0) gives (N/2,).
    
    grad_phases = (2.0 / N) * np.sum(sin_terms * (grad_h[:, None] * mags[None, :]), axis=0)
    
    return np.concatenate([grad_mags, grad_phases])

def _refine_spatial(h, deadline, beta_start=30.0, beta_end=150.0):
    """Refine a candidate h using spatial L-BFGS-B."""
    h_curr = h.copy()
    remaining = deadline - time.time()
    if remaining < 5.0:
        return h_curr
    
    # Phase 1: Moderate beta
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(beta_start,), jac=_objective_soft_grad,
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 2: Higher beta
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(beta_end,), jac=_objective_soft_grad,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9})
            h_curr = res.x
        except Exception:
            pass
            
    return h_curr

def _refine_spectral(h0, deadline):
    """Refine a candidate h using spectral L-BFGS-B."""
    # Extract spectral params from h0
    H = np.fft.rfft(h0)
    # Normalize DC to 1 to fix the scale (since objective is scale invariant)
    # Actually, if we change scale, value doesn't change.
    # But to keep params bounded, we fix H[0]=1.
    # h = ifft(H). If we scale H by c, h scales by c.
    # Value is invariant to scaling.
    # So we can fix H[0] = 1.
    
    # Current H[0] might not be 1.
    # Let's just use the current H, but fix H[0] to its current real value?
    # No, let's fix H[0]=1 and scale the initial mags/phases accordingly.
    # Or simpler: just optimize mags/phases for k>=1, and let H[0] be free?
    # If H[0] is free, scaling is a flat direction. L-BFGS-B might struggle.
    # Let's fix H[0]=1.
    
    # We need to map h0 to params such that _spectral_to_h(params) is close to h0.
    # h0 = ifft(H0).
    # If we set H_new[0]=1, and H_new[k] = H0[k]/H0[0], then h_new = h0 / H0[0].
    # Since value is scale invariant, this is fine.
    
    H0 = H.copy()
    if abs(H0[0]) < 1e-10:
        return h0 # Degenerate case
        
    H_norm = H0 / H0[0]
    
    mags = np.abs(H_norm[1:])
    phases = np.angle(H_norm[1:])
    
    init_params = np.concatenate([mags, phases])
    
    h_curr = h0.copy()
    
    # Optimize in spectral domain
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_spectral_objective, init_params, method='L-BFGS-B',
                           args=(50.0,), jac=_spectral_grad,
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.fun < _objective_soft(h_curr, 50.0):
                mags_opt = res.x[:N//2]
                phases_opt = res.x[N//2:]
                h_opt = _spectral_to_h(mags_opt, phases_opt)
                # Scale back? No, scale doesn't matter.
                # But we should ensure h is real and finite.
                if np.all(np.isfinite(h_opt)):
                    h_curr = h_opt
        except Exception:
            pass
    
    # Polish in spatial domain
    if time.time() < deadline - 5.0:
        h_curr = _refine_spatial(h_curr, deadline, beta_start=50.0, beta_end=100.0)
        
    return h_curr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    i = np.arange(N)
    
    # 1. Seed
    initializations.append(np.ones(N))
    
    # 2. Low-frequency sinusoids with DC offset
    for k in [1, 2, 3, 4]:
        for a in [0.5, 1.0, 2.0]:
            h0 = a + np.sin(2*np.pi*k*i/N)
            initializations.append(h0)
            
    # 3. Comb patterns
    for period in [2, 3, 4, 5, 8, 10]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) > 1.0:
            initializations.append(h0)
            
    # 4. Random smooth functions
    for _ in range(5):
        h0 = np.sin(2*np.pi*rng.uniform(0.1, 2.0)*i/N + rng.uniform(0, 2*np.pi))
        h0 += 1.0
        initializations.append(h0)
        
    # 5. Spectral Initializations
    # Flat spectrum (random phases, constant magnitude)
    for _ in range(5):
        mags = np.ones(N//2)
        phases = rng.uniform(0, 2*np.pi, N//2)
        h0 = _spectral_to_h(mags, phases)
        # Add DC
        h0 += 1.0
        initializations.append(h0)
        
    # Peaked spectrum (high low-freq, low high-freq)
    for _ in range(5):
        mags = np.exp(-np.linspace(0, 3, N//2))
        phases = rng.uniform(0, 2*np.pi, N//2)
        h0 = _spectral_to_h(mags, phases)
        h0 += 1.0
        initializations.append(h0)

    # Optimize each initialization
    for h0 in initializations:
        if time.time() > deadline - 30.0:
            break
            
        # Try both spatial and spectral refinement
        h_refined_sp = _refine_spatial(h0, deadline)
        val_sp = _compute_value(h_refined_sp)
        
        if val_sp < best_val:
            best_val = val_sp
            best_h = h_refined_sp.copy()
            
        if time.time() > deadline - 15.0:
            continue
            
        h_refined_spec = _refine_spectral(h0, deadline)
        val_spec = _compute_value(h_refined_spec)
        
        if val_spec < best_val:
            best_val = val_spec
            best_h = h_refined_spec.copy()
            
    # Final polish on the best found
    if time.time() < deadline - 5.0:
        h_final = _refine_spatial(best_h, deadline, beta_start=100.0, beta_end=200.0)
        val_final = _compute_value(h_final)
        if val_final < best_val:
            best_h = h_final
            best_val = val_final

    s_final = np.sum(best_h)
    if abs(s_final) < 1e-10:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
