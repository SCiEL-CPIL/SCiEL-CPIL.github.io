"""Convex QP-based autoconvolution peak suppression with smooth refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 142.0  # Safety margin before 150s kill

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _build_conv_matrix():
    """
    Build the matrix M such that c_k = sum_{i,j: i+j=k} h_i * h_j = (M @ h)_k
    where c = convolve(h, h).
    M is (2N-1) x N.
    """
    M = np.zeros((2 * N - 1, N))
    for i in range(N):
        for j in range(N):
            M[i + j, i] += 1.0
    # Actually, for autoconvolution: c_k = sum_{i=0}^{k} h_i * h_{k-i} for k < N
    # and c_k = sum_{i=k-N+1}^{N-1} h_i * h_{k-i} for k >= N
    # The quadratic form is h^T Q_k h where Q_k has Q_k[i,j] = 1 if i+j=k
    # We need M such that c = M @ h where M[i,j] contributes to c_{i+j}
    # But c_k = sum_{i+j=k} h_i h_j, which is quadratic, not linear in h.
    # So we can't represent c as M @ h directly.
    # Instead, we note that c_k = sum_i h_i * (h convolved with delta at position k-i)
    # Let's use the Toeplitz structure: c = C(h) * h where C(h) is a Toeplitz-like matrix
    # Actually, c_k = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h_i * h_{k-i}
    # This is quadratic. For the QP approach, we minimize ||c||^2 = sum_k c_k^2
    # which is a quartic in h. That's not a QP.
    # 
    # Alternative: minimize max_k c_k subject to sum h = 1.
    # This is not convex in general because c_k is quadratic.
    #
    # Let's use the smooth surrogate approach more effectively.
    # The key insight: if we want to minimize the peak of autoconvolution,
    # we should make h have oscillatory structure so that h_i * h_j tends to cancel.
    #
    # Better approach: use the fact that in Fourier domain,
    # conv(h,h) corresponds to |H|^2 in frequency domain (up to scaling).
    # So we want to minimize the inverse FFT peak of |H|^2.
    #
    # Let's try a different strategy: parameterize h via its DFT coefficients
    # and optimize for flat |H|^2 spectrum (which gives flat autoconvolution).
    
    # For now, let's just build the linear operator for the gradient computation
    # and use the smooth surrogate directly.
    return None

def _autoconv(h):
    """Compute autoconvolution efficiently."""
    return np.convolve(h, h)

def _smooth_objective(h, beta=100.0):
    """
    Smooth surrogate: replace max(conv) with log-sum-exp approximation.
    """
    h = np.asarray(h, dtype=np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv(h)
    max_c = float(np.max(conv))
    # Log-sum-exp for stability
    shifted = conv - max_c
    # Avoid overflow
    max_shifted = np.max(shifted)
    if max_shifted > 50:
        shifted = shifted - max_shifted
        max_c_adj = max_c + max_shifted
    else:
        max_c_adj = max_c
    log_sum = max_c_adj + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = abs(2.0 * N * log_sum / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_grad(h, beta=100.0):
    """
    Gradient of the smooth surrogate.
    """
    h = np.asarray(h, dtype=np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _autoconv(h)
    max_c = float(np.max(conv))
    shifted = conv - max_c
    max_shifted = np.max(shifted)
    if max_shifted > 50:
        shifted = shifted - max_shifted
        max_c_adj = max_c + max_shifted
    else:
        max_c_adj = max_c
    
    # Soft-max weights
    exp_vals = np.exp(beta * shifted)
    weight_sum = np.sum(exp_vals)
    weights = exp_vals / weight_sum
    
    # lse = max_c_adj + log(weight_sum) / beta
    lse = max_c_adj + np.log(weight_sum) / beta
    
    # Gradient of lse w.r.t. h_i:
    # lse ≈ max_c + (1/beta) * log(sum_k exp(beta * (c_k - max_c)))
    # d lse / d h_i = sum_k weights_k * d c_k / d h_i
    # d c_k / d h_i = sum_j (delta_{i+j,k} h_j + delta_{j+i,k} h_j) = 2 * (h convolved with delta at k-i)
    # More precisely: c_k = sum_{j: j in [0,N-1], k-j in [0,N-1]} h_j * h_{k-j}
    # d c_k / d h_i = sum_{j: k-j=i, j valid} h_j + sum_{j: j=i, k-j valid} h_{k-i}
    #                = h_{k-i} (if k-i in [0,N-1]) + h_{k-i} (if k-i in [0,N-1])
    # Wait, let me be more careful.
    # c_k = sum_{i=0}^{N-1} sum_{j=0}^{N-1} h_i h_j * 1[i+j=k]
    # d c_k / d h_m = sum_{i,j} [delta_{i,m} h_j + h_i delta_{j,m}] * 1[i+j=k]
    #               = sum_{j: m+j=k} h_j + sum_{i: i+m=k} h_i
    #               = h_{k-m} (if 0 <= k-m <= N-1) + h_{k-m} (if 0 <= k-m <= N-1)
    #               = 2 * h_{k-m} (if 0 <= k-m <= N-1)
    # But wait, this double counts when i=j=m, i.e., when k=2m.
    # Actually no: the two sums are over different indices.
    # First sum: i=m, so j=k-m. Contribution: h_{k-m} if 0<=k-m<N.
    # Second sum: j=m, so i=k-m. Contribution: h_{k-m} if 0<=k-m<N.
    # So d c_k / d h_m = 2 * h_{k-m} if 0 <= k-m <= N-1, else 0.
    # But when k-m = m (i.e., k=2m), we're counting the term h_m * h_m twice,
    # which is correct because d(h_m^2)/dh_m = 2*h_m.
    
    # So d lse / d h_m = sum_k weights_k * 2 * h_{k-m} * 1[0<=k-m<N]
    #                  = 2 * sum_k weights_k * h_{k-m} (where k-m is in [0,N-1])
    
    # Let's compute this efficiently.
    # For each m, d_lse_dh[m] = 2 * sum_{k: 0<=k-m<N} weights[k] * h[k-m]
    # Let j = k-m, then k = m+j, j in [0,N-1], k in [m, m+N-1]
    # d_lse_dh[m] = 2 * sum_{j=0}^{N-1} weights[m+j] * h[j] (where m+j < 2N-1)
    
    # This is a cross-correlation of weights and h.
    # Use FFT for efficiency.
    from scipy.signal import fftconvolve
    
    # d_lse_dh = 2 * cross-correlation of weights and h
    # cross_corr[m] = sum_j weights[m+j] * h[j]
    # This is the same as (weights flipped) convolved with h, shifted appropriately.
    
    # Let's compute via FFT
    n_fft = 2 * (2 * N - 1)
    W = np.fft.rfft(weights, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    # cross_corr[m] = sum_j weights[m+j] h[j]
    # = sum_j h[j] weights[m+j]
    # Let's define w_rev[k] = weights[2N-2-k] (reverse weights)
    # Then cross_corr = ifft(fft(h) * conj(fft(w_rev)))... 
    # Actually, let's just use direct computation since N=400 is small.
    
    d_lse_dh = np.zeros(N)
    for m in range(N):
        # d_lse_dh[m] = 2 * sum_{j=0}^{N-1} weights[m+j] * h[j] (if m+j < 2N-1)
        valid_j = np.arange(N)
        k_vals = m + valid_j
        mask = k_vals < (2 * N - 1)
        d_lse_dh[m] = 2.0 * np.sum(weights[k_vals[mask]] * h[valid_j[mask]])
    
    # Now, f = 2N * lse / s^2
    # df/dh_m = 2N * (d_lse_dh[m] * s^2 - lse * 2 * s * 1) / s^4
    #         = 2N * (d_lse_dh[m] * s - 2 * lse) / s^3
    
    f = 2.0 * N * lse / (s * s)
    df_dh = 2.0 * N * (d_lse_dh * s - 2.0 * lse) / (s * s * s)
    
    # The objective is |f|, so:
    if f >= 0:
        grad = df_dh
    else:
        grad = -df_dh
    
    return grad

def _fourier_flat_init(freqs=None):
    """
    Generate h with approximately flat |H|^2 spectrum to flatten autoconvolution.
    """
    if freqs is None:
        rng = np.random.RandomState(42)
        freqs = sorted(rng.choice(np.arange(1, N//2), size=min(20, N//2-1), replace=False))
    
    h = np.zeros(N)
    for f in freqs:
        phase = np.random.RandomState(f).uniform(0, 2*np.pi)
        h += np.cos(2 * np.pi * f * np.arange(N) / N + phase)
        h += 0.5 * np.sin(2 * np.pi * f * np.arange(N) / N + phase)
    
    # Normalize
    s = np.sum(h)
    if abs(s) < 1e-10:
        h += 0.1
        s = np.sum(h)
    h = h / s  # Make sum = 1
    
    return h

def _multi_freq_cosine_init():
    """Generate diverse multi-frequency cosine initializations."""
    inits = []
    i_arr = np.arange(N)
    t = i_arr / N
    
    # Various frequency combinations
    freq_combos = [
        [1, 3, 5, 7],
        [2, 5, 8, 11],
        [1, 4, 7, 10, 13],
        [3, 6, 9, 12],
        [1, 2, 3, 4, 5, 6, 7],
        [2, 4, 6, 8, 10],
        [1, 3, 5, 7, 9, 11],
        [5, 10, 15, 20],
        [1, 5, 10, 15, 25],
        [2, 7, 12, 17],
    ]
    
    for freqs in freq_combos:
        h = np.zeros(N)
        for f in freqs:
            h += np.cos(2 * np.pi * f * t)
            h += 0.3 * np.sin(2 * np.pi * f * t)
        
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h)
    
    # Also try with alternating signs
    for freqs in freq_combos[:5]:
        h = np.zeros(N)
        for i, f in enumerate(freqs):
            sign = 1 if i % 2 == 0 else -1
            h += sign * np.cos(2 * np.pi * f * t)
        
        s = np.sum(h)
        if abs(s) < 1e-10:
            h += 0.05
        inits.append(h)
    
    return inits

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    # --- Phase 1: Generate diverse initializations ---
    candidates = []
    
    # Fourier-flat initialization
    for seed in range(5):
        rng = np.random.RandomState(seed)
        freqs = sorted(rng.choice(np.arange(1, N//2), size=min(15, N//2-1), replace=False))
        h = _fourier_flat_init(freqs)
        candidates.append(h)
    
    # Multi-frequency cosine initializations
    candidates.extend(_multi_freq_cosine_init())
    
    # Random smooth perturbations
    rng = np.random.RandomState(42)
    for _ in range(15):
        h_r = rng.uniform(-1, 1, N)
        from scipy.ndimage import gaussian_filter1d
        h_r = gaussian_filter1d(h_r, sigma=rng.uniform(5, 30))
        s = np.sum(h_r)
        if abs(s) < 1e-10:
            h_r += 0.05
        candidates.append(h_r)
    
    # Evaluate all candidates
    cand_vals = []
    for h in candidates:
        v = _eval_value(h)
        if np.isfinite(v):
            cand_vals.append((v, h))
    
    cand_vals.sort(key=lambda x: x[0])
    
    # Keep top 15
    top_candidates = [h for _, h in cand_vals[:15]]
    
    # Update best
    if top_candidates:
        v_top = _eval_value(top_candidates[0])
        if v_top < best_val:
            best_val = v_top
            best_h = top_candidates[0].copy()
    
    # --- Phase 2: Smooth surrogate optimization from top initializations ---
    for h_init in top_candidates[:10]:
        if elapsed() > DEADLINE - 30:
            break
        
        s = np.sum(h_init)
        if abs(s) < 1e-10:
            h_init = h_init + 0.05
        
        for beta in [50.0, 100.0, 200.0]:
            if elapsed() > DEADLINE - 10:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    h_init,
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 500, 'gtol': 1e-10}
                )
                
                if elapsed() > DEADLINE - 5:
                    break
                
                h_candidate = res.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
                    h_init = best_h.copy()
                    
            except Exception:
                pass
    
    # --- Phase 3: High-precision polish ---
    if elapsed() < DEADLINE - 15:
        h0 = best_h.copy()
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            h0 = h0 + 0.01
        
        for beta in [200.0, 400.0, 800.0]:
            if elapsed() > DEADLINE - 5:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    h0,
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 800, 'gtol': 1e-12}
                )
                
                if elapsed() > DEADLINE - 3:
                    break
                
                h_candidate = res.x
                v = _eval_value(h_candidate)
                if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    best_val = v
                    best_h = h_candidate.copy()
                    h0 = best_h.copy()
                    
            except Exception:
                pass
    
    # --- Phase 4: Try a few more diverse starts if time permits ---
    if elapsed() < DEADLINE - 20:
        # Try structured patterns
        i_arr = np.arange(N)
        t = i_arr / N
        
        extra_inits = []
        
        # Wigner-like functions
        for a in [0.5, 1.0, 2.0, 3.0]:
            for b in [0.5, 1.0, 2.0, 3.0]:
                h = np.exp(-a * (t - 0.5)**2) * np.cos(2 * np.pi * b * t)
                extra_inits.append(h)
        
        # Chirp signals
        for alpha in [2, 5, 10]:
            h = np.cos(2 * np.pi * (t + alpha * t**2))
            extra_inits.append(h)
            extra_inits.append(-h)
        
        # Evaluate and refine best
        for h_init in extra_inits:
            if elapsed() > DEADLINE - 10:
                break
            
            s = np.sum(h_init)
            if abs(s) < 1e-10:
                h_init = h_init + 0.05
            
            v_init = _eval_value(h_init)
            if v_init < best_val + 0.5:  # Only refine if promising
                try:
                    res = minimize(
                        lambda x: _smooth_objective(x, 100.0),
                        h_init,
                        jac=lambda x: _smooth_grad(x, 100.0),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-10}
                    )
                    
                    if elapsed() > DEADLINE - 5:
                        break
                    
                    h_candidate = res.x
                    v = _eval_value(h_candidate)
                    if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                        best_val = v
                        best_h = h_candidate.copy()
                        
                except Exception:
                    pass
    
    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
