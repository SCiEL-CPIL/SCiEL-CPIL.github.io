"""Structural approach: Fourier-domain construction + direct iterative refinement.

Key insight: value = 2*N*max(conv)/sum(h)^2 >= 2 by Cauchy-Schwarz.
We minimize by:
1. Constructing h in Fourier domain to have strong DC component and spread high-freq energy
2. Directly iterating to reduce the maximum autocorrelation value
3. Using a different optimization family: projected subgradient on the max constraint
"""
import numpy as np
import time

N = 400
_DEADLINE = 145.0


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _fourier_construct(alpha=0.5, n_peaks=5):
    """Construct h via Fourier synthesis: strong DC + spread high-frequency energy."""
    # Start with DC-dominant spectrum
    H = np.zeros(N, dtype=complex)
    # Strong DC
    H[0] = 1.0
    # Add spread high-frequency components with controlled amplitude
    # Use symmetric peaks to keep h real
    freqs = np.array([N//4, N//3, N//5, N//6, N//8, N//10, N//12, N//15, N//20, N//25])
    freqs = freqs[freqs < N//2]
    for f in freqs[:n_peaks]:
        H[f] = alpha * (0.5 + 0.5 * np.cos(2 * np.pi * f / N))
        H[N - f] = np.conj(H[f])
    
    h = np.real(np.fft.ifft(H))
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.ones(N)
    return h / s


def _iterative_refine(h0, max_iter=5000, max_time=30.0):
    """Direct iterative refinement: push down the max autocorrelation value.
    
    Uses a projected subgradient approach on the constraint max(conv) <= target.
    """
    t0 = time.time()
    h = h0.copy()
    best_h = h.copy()
    best_val = _value(h)
    
    # Normalize to sum=1
    s = np.sum(h)
    if abs(s) < 1e-12:
        return h, best_val
    h = h / s
    
    # Target: reduce max conv below current level
    conv = np.convolve(h, h)
    current_max = np.max(conv)
    target = current_max * 0.99  # Try to reduce by 1%
    
    for it in range(max_iter):
        if time.time() - t0 > max_time:
            break
        if it % 100 == 0 and time.time() - t0 > max_time * 0.8:
            break
        
        # Compute autocorrelation
        conv = np.convolve(h, h)
        max_val = np.max(conv)
        s = np.sum(h)
        if abs(s) < 1e-12:
            break
        
        # Find which lag(s) are dominant
        # The gradient of max(conv) w.r.t. h is related to the structure of conv
        # For the max at lag k*, d(conv[k*])/d(h[j]) = 2*h[k*-j] (valid range)
        
        # Use subgradient: if max is at lag k*, push h to reduce conv[k*]
        # conv[k] = sum_i h[i]*h[i+k], so to reduce it, we want to make
        # h[i] and h[i+k] have opposite signs or be small
        
        # Simple approach: identify the peak lag and apply a corrective update
        peak_lag = np.argmax(conv)
        
        # Gradient of conv[peak_lag] w.r.t. h:
        # d(conv[peak_lag])/d(h[j]) = 2*h[peak_lag - j] if 0 <= peak_lag-j < N
        #                            + 2*h[peak_lag - j + N] if peak_lag-j >= N (not applicable for full conv)
        # For full conv of length 2N-1, conv[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k] for k>=0
        # and conv[k] = sum_{i=0}^{k} h[i]*h[k-i] ... actually let's use the standard formula
        
        # d(conv[k])/d(h[j]) = h[k-j] + h[k-j+N] (with appropriate bounds)
        # More carefully: conv[k] = sum_{i} h[i] * h[i+k] where i and i+k in [0,N-1]
        # d(conv[k])/d(h[j]) = h[j+k] (if j+k < N) + h[j-k] (if j-k >= 0)
        # = sum over valid terms
        
        grad_conv = np.zeros(N)
        if peak_lag < N:
            # For lag k < N: conv[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
            # d(conv[k])/d(h[j]) = h[j+k] (if j+k < N, i.e., j < N-k) 
            #                     + h[j-k] (if j-k >= 0, i.e., j >= k)
            for j in range(N):
                g = 0.0
                if j + peak_lag < N:
                    g += h[j + peak_lag]
                if j - peak_lag >= 0:
                    g += h[j - peak_lag]
                grad_conv[j] = 2.0 * g
        else:
            # For lag k >= N: conv[k] = sum_{i=k-N+1}^{N-1} h[i]*h[i-(k-N+1)]
            # Let m = k - N + 1, then conv[k] = sum_{i=m}^{N-1} h[i]*h[i-m]
            # d(conv[k])/d(h[j]) = h[j-m] (if j-m >= 0 and j >= m) 
            #                     + h[j+m] (if j+m < N and j < N-m)
            m = peak_lag - N + 1
            for j in range(N):
                g = 0.0
                if j - m >= 0 and j - m < N:
                    g += h[j - m]
                if j + m >= 0 and j + m < N:
                    g += h[j + m]
                grad_conv[j] = 2.0 * g
        
        # Also account for the 1/s^2 factor in the objective
        # f = 2N * max_conv / s^2
        # df/dh = 2N * (grad_conv / s^2 - 2*max_conv * grad_s / s^3)
        # where grad_s = ones (since s = sum h)
        grad_obj = 2.0 * N * (grad_conv / (s * s) - 2.0 * max_val / (s * s * s))
        
        # Subgradient step
        # Normalize gradient
        g_norm = np.linalg.norm(grad_obj)
        if g_norm < 1e-12:
            break
        grad_norm = grad_obj / g_norm
        
        # Adaptive step size
        step = 0.01 / (1.0 + 0.1 * it)
        
        h_new = h - step * grad_norm
        
        # Project back to sum=1
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-12:
            break
        h_new = h_new / s_new
        
        # Check improvement
        val_new = _value(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
        
        h = h_new
    
    return best_h, best_val


def _spectral_design(rng_seed=42):
    """Design h by directly optimizing the spectrum to minimize peak autocorrelation."""
    rng = np.random.default_rng(rng_seed)
    
    # Start with a random spectrum
    H = np.zeros(N, dtype=complex)
    H[0] = 1.0  # DC
    
    # Add random phases to high frequencies
    for f in range(1, N//2):
        amp = 0.1 / np.sqrt(f + 1)  # Decay with frequency
        phase = rng.uniform(0, 2 * np.pi)
        H[f] = amp * np.exp(1j * phase)
        H[N - f] = np.conj(H[f])
    
    h = np.real(np.fft.ifft(H))
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.ones(N)
    return h / s


def _direct_max_reduction(h0, n_rounds=100, max_time=20.0):
    """Directly reduce the maximum autocorrelation by identifying and suppressing peaks."""
    t0 = time.time()
    h = h0.copy()
    best_h = h.copy()
    best_val = _value(h)
    
    s = np.sum(h)
    if abs(s) < 1e-12:
        return h, best_val
    h = h / s
    
    for round_idx in range(n_rounds):
        if time.time() - t0 > max_time:
            break
        
        conv = np.convolve(h, h)
        max_val = np.max(conv)
        peak_idx = np.argmax(conv)
        
        # If max is at lag 0, we can't easily reduce it without changing sum(h^2)
        # Focus on reducing off-peak values
        if peak_idx == 0:
            # Try to spread energy more evenly
            # Apply a small diffusion
            h_smooth = np.convolve(h, np.ones(5)/5, mode='same')
            s_sm = np.sum(h_smooth)
            if abs(s_sm) < 1e-12:
                break
            h_smooth = h_smooth / s_sm
            val_sm = _value(h_smooth)
            if val_sm < best_val:
                best_val = val_sm
                best_h = h_smooth.copy()
                h = h_smooth
            continue
        
        # For off-peak max, try to reduce it
        # The contribution to conv[peak_idx] comes from pairs (i, i+peak_idx)
        # We want to make these products smaller
        
        # Simple heuristic: identify the elements contributing most to the peak
        # and reduce their magnitudes
        
        if peak_idx < N:
            # conv[peak_idx] = sum_{i=0}^{N-1-peak_idx} h[i]*h[i+peak_idx]
            # Find the term with largest absolute contribution
            contributions = np.array([h[i] * h[i + peak_idx] for i in range(N - peak_idx)])
            if len(contributions) > 0:
                max_contrib_idx = np.argmax(np.abs(contributions))
                i_star = max_contrib_idx
                # Reduce h[i_star] or h[i_star + peak_idx]
                # Choose the one with larger magnitude
                if abs(h[i_star]) >= abs(h[i_star + peak_idx]):
                    h[i_star] *= 0.95
                else:
                    h[i_star + peak_idx] *= 0.95
        else:
            m = peak_idx - N + 1
            contributions = np.array([h[i] * h[i - m] for i in range(m, N)])
            if len(contributions) > 0:
                max_contrib_idx = np.argmax(np.abs(contributions))
                i_star = m + max_contrib_idx
                if abs(h[i_star]) >= abs(h[i_star - m]):
                    h[i_star] *= 0.95
                else:
                    h[i_star - m] *= 0.95
        
        # Renormalize
        s = np.sum(h)
        if abs(s) < 1e-12:
            break
        h = h / s
        
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val


def _construct_baseline():
    """Baseline: uniform gives value 2.0."""
    return np.ones(N)


def construct():
    t0 = time.time()
    
    def time_left():
        return _DEADLINE - (time.time() - t0)
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Fourier-based constructions
    if time_left() > 60:
        # Try different Fourier constructions
        for alpha in [0.3, 0.5, 0.7, 1.0]:
            for n_peaks in [3, 5, 8, 12]:
                if time_left() < 10:
                    break
                h = _fourier_construct(alpha=alpha, n_peaks=n_peaks)
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
        
        # Spectral design
        for seed in range(10):
            if time_left() < 10:
                break
            h = _spectral_design(rng_seed=seed)
            v = _value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # Phase 2: Iterative refinement on best candidate
    if time_left() > 30:
        h_ref, val_ref = _iterative_refine(best_h, max_iter=3000, max_time=min(25.0, time_left() - 15.0))
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 3: Direct max reduction
    if time_left() > 15:
        h_dir, val_dir = _direct_max_reduction(best_h, n_rounds=200, max_time=min(12.0, time_left() - 8.0))
        if val_dir < best_val:
            best_val = val_dir
            best_h = h_dir
    
    # Phase 4: Final polish with iterative refinement
    if time_left() > 10:
        h_pol, val_pol = _iterative_refine(best_h, max_iter=2000, max_time=min(8.0, time_left() - 5.0))
        if val_pol < best_val:
            best_val = val_pol
            best_h = h_pol
    
    # Safety checks
    best_h = np.asarray(best_h, dtype=float)
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        best_h = np.ones(N)
    
    if _value(best_h) > 2.0 + 1e-9:
        best_h = np.ones(N)
    
    return best_h


# Self-check
if __name__ == "__main__":
    _h = construct()
    _s = _h.sum()
    _conv = np.convolve(_h, _h, mode="full")
    print("value =", abs(2.0 * N * np.max(_conv) / (_s * _s)),
          "sum =", _s, "finite =", bool(np.all(np.isfinite(_h))))
