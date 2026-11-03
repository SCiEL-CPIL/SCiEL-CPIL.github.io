"""Search for a low-autocorrelation signed step function (400 equal intervals).

Goal: minimise the peak aperiodic autocorrelation sidelobe of a signed (+-1)
sequence of length 400; this drives the C3 autoconvolution upper bound below
the all-ones baseline.

This version uses:
  1.  Continuous relaxation with soft-max (log-sum-exp) surrogate for the
      peak sidelobe, optimized via gradient descent in R^N.
  2.  Multiple restarts with different initializations and seeds.
  3.  Greedy single-flip and pair-flip polish with incremental AC updates.
  4.  Deadline guard with safety margin.
"""
import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _full_ac(h):
    """Compute full aperiodic autocorrelation array (length N)."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_from_ac(ac):
    """Peak sidelobe from an existing AC array."""
    return float(np.max(np.abs(ac[1:])))


# ---------------------------------------------------------------------------
# Continuous relaxation with soft-max surrogate
# ---------------------------------------------------------------------------

def _round_to_sign(h):
    """Round continuous values to ±1."""
    return np.where(h >= 0, 1.0, -1.0)


def _soft_max_grad_ac(ac, tau):
    """
    Compute gradient of soft-max objective wrt ac.
    Returns (obj, grad_ac) where grad_ac[k] = d(obj)/d(ac[k]) for k=1..N-1.
    """
    abs_ac = np.abs(ac[1:])
    m = np.max(abs_ac)
    shifted = tau * (abs_ac - m)
    shifted = np.clip(shifted, -700, 700)
    exp_shifted = np.exp(shifted)
    Z = np.sum(exp_shifted)
    obj = m + np.log(Z)
    grad_ac = np.zeros(N)
    signs = np.sign(ac[1:])
    grad_ac[1:] = signs * exp_shifted / Z
    return obj, grad_ac


def _compute_grad_h(h, grad_ac):
    """
    Compute gradient of soft-max objective wrt h, given grad_ac.
    Uses FFT for O(N log N) instead of O(N^2).
    """
    g = grad_ac[1:]  # length N-1
    # grad_h[i] = sum_{k=1}^{N-1} g[k] * (h[i+k] if i+k<N) + g[k] * (h[i-k] if i-k>=0)
    # First term: for each i, sum over k where i+k < N: g[k]*h[i+k]
    # This is a cross-correlation of g and h (truncated)
    # Second term: for each i, sum over k where i-k >= 0: g[k]*h[i-k]
    # This is a convolution of g and h (truncated)
    
    # Use FFT-based approach
    # Build full correlation: corr[m] = sum_k g[k] * h[m+k] for valid m
    # And conv[m] = sum_k g[k] * h[m-k] for valid m
    
    # For efficiency, pad to next power of 2
    size = 1
    while size < 2 * N:
        size *= 2
    
    H = np.fft.fft(h, size)
    G = np.fft.fft(g, size)
    
    # First term: grad_h[i] += sum_{k=1}^{N-1-i} g[k] * h[i+k]
    # This is like a "forward" correlation
    # corr_forward[i] = sum_k g[k] * h[i+k]
    # = sum_k h[i+k] * g[k]
    # In FFT: this is ifft(conj(H) * G) but we need to be careful with indexing
    # Actually, let's think of it as: for fixed i, sum over k of g[k]*h[i+k]
    # This is the cross-correlation of h and g evaluated at lag -i
    # cross_corr[lag] = sum_j h[j] * g[j-lag] (with appropriate bounds)
    
    # Simpler: use the fact that both terms are linear operations
    # Term 1: T1[i] = sum_{k=1}^{N-1} g[k] * h[i+k] * 1[i+k < N]
    # Term 2: T2[i] = sum_{k=1}^{N-1} g[k] * h[i-k] * 1[i-k >= 0]
    
    # T2[i] = sum_{k=1}^{i} g[k] * h[i-k] = (g * h)[i] for i >= 1 (convolution)
    # T1[i] = sum_{k=1}^{N-1-i} g[k] * h[i+k]
    # Let j = i+k, so k = j-i, j ranges from i+1 to N-1
    # T1[i] = sum_{j=i+1}^{N-1} g[j-i] * h[j]
    # This is like a reversed convolution
    
    # Let's just compute both via FFT
    # T2 is standard convolution of g and h
    conv = np.fft.ifft(H * G).real[:N]
    # conv[i] = sum_{k=0}^{i} h[k] * g[i-k] for i < N
    # But we want sum_{k=1}^{i} g[k] * h[i-k] = sum_{k=1}^{i} h[i-k] * g[k]
    # conv[i] = h[0]*g[i] + h[1]*g[i-1] + ... + h[i]*g[0]
    # We want: h[i-1]*g[1] + h[i-2]*g[2] + ... + h[0]*g[i]
    # That's conv[i] - h[0]*g[i] - h[i]*g[0]... no wait
    # conv[i] = sum_{k=0}^{min(i,N-1)} h[k]*g[i-k]
    # We want sum_{k=1}^{i} h[i-k]*g[k] = sum_{j=0}^{i-1} h[j]*g[i-j] where j=i-k
    # = conv[i] - h[0]*g[i] - h[i]*g[0]... 
    # Actually conv[i] = h[0]*g[i] + h[1]*g[i-1] + ... + h[i-1]*g[1] + h[i]*g[0]
    # We want h[0]*g[i] + h[1]*g[i-1] + ... + h[i-1]*g[1] = conv[i] - h[i]*g[0]
    # But g[0] is not defined (g starts at index 0 which is grad_ac[1])
    # Wait, g = grad_ac[1:], so g[0] = grad_ac[1], g[1] = grad_ac[2], etc.
    # So in the convolution, g[k] refers to grad_ac[k+1]
    
    # Let me redefine: let f[k] = grad_ac[k+1] for k=0..N-2
    # T2[i] = sum_{k=1}^{i} f[k-1] * h[i-k] = sum_{j=0}^{i-1} f[j] * h[i-1-j]
    # This is (f * h)[i-1] where * denotes convolution
    # conv_fh[m] = sum_{j=0}^{m} f[j] * h[m-j]
    # T2[i] = conv_fh[i-1] for i >= 1
    
    # T1[i] = sum_{k=1}^{N-1-i} f[k-1] * h[i+k]
    # Let j = i+k, k = j-i, j from i+1 to N-1
    # T1[i] = sum_{j=i+1}^{N-1} f[j-i-1] * h[j]
    # Let m = j-i-1, j = i+1+m, m from 0 to N-2-i
    # T1[i] = sum_{m=0}^{N-2-i} f[m] * h[i+1+m]
    # This is a cross-correlation: sum_m f[m] * h[i+1+m]
    # = (f cross-corr h)[i+1] where cross-corr at lag L = sum_m f[m]*h[m+L]
    
    # Cross-correlation via FFT: corr[L] = sum_m f[m]*h[m+L] = ifft(conj(FFT(f)) * FFT(h))[L]
    # But we need to handle the truncation carefully
    
    # Let's just do it with explicit FFT
    F = np.fft.fft(f, size)
    H_full = np.fft.fft(h, size)
    
    # T2[i] = sum_{j=0}^{i-1} f[j]*h[i-1-j] = conv[f,h][i-1]
    conv_fh = np.fft.ifft(F * H_full).real[:N]
    
    # T1[i] = sum_{m=0}^{N-2-i} f[m]*h[i+1+m]
    # This is the cross-correlation of f and h at lag (i+1)
    # corr[L] = sum_m f[m]*h[m+L]
    # In FFT: corr = ifft(conj(F) * H_full)
    # But this gives the circular correlation. For linear, we need padding.
    # With size >= 2N, the linear correlation is embedded.
    corr = np.fft.ifft(np.conj(F) * H_full).real
    # corr[L] for L=0..size-1
    # For L in 0..N-1: corr[L] = sum_{m=0}^{N-1-L} f[m]*h[m+L] (approximately, with padding)
    # Actually with zero-padding to size >= 2N, the linear cross-correlation is:
    # corr_linear[L] = sum_m f[m]*h[m+L] for m such that both indices valid
    # The FFT gives us the circular version, but with sufficient padding it equals linear
    
    # T1[i] = corr[i+1] for i+1 <= N-1, i.e., i <= N-2
    # For i = N-1, T1[N-1] = 0
    
    T2 = np.zeros(N)
    for i in range(1, N):
        T2[i] = conv_fh[i-1]
    
    T1 = np.zeros(N)
    for i in range(N-1):
        T1[i] = corr[i+1]
    
    grad_h = T1 + T2
    return grad_h


def _continuous_relax(h_init, time_budget, start, seed=0):
    """
    Continuous gradient descent on soft-max objective.
    Returns best ±1 sequence found.
    """
    rng = np.random.default_rng(seed)
    h = h_init.astype(np.float64).copy()
    tau = 1.0
    lr = 0.5
    best_h = _round_to_sign(h)
    best_val = _peak_sidelobe(best_h)
    
    step = 0
    while time.time() - start < time_budget:
        ac = _full_ac(h)
        obj, grad_ac = _soft_max_grad_ac(ac, tau)
        grad_h = _compute_grad_h(h, grad_ac)
        
        # Normalize gradient for stability
        gnorm = np.linalg.norm(grad_h)
        if gnorm > 1e-10:
            h = h - lr * grad_h / gnorm * np.linalg.norm(h)
        else:
            h = h - lr * grad_h
        
        if step % 3 == 0:
            h_rounded = _round_to_sign(h)
            val = _peak_sidelobe(h_rounded)
            if val < best_val:
                best_val = val
                best_h = h_rounded.copy()
        
        tau *= 1.03
        lr *= 0.999
        
        if step % 30 == 0 and rng.random() < 0.4:
            h = _round_to_sign(h) + 0.3 * rng.randn(N)
        
        step += 1
    
    h_final = _round_to_sign(h)
    val_final = _peak_sidelobe(h_final)
    if val_final < best_val:
        best_val = val_final
        best_h = h_final.copy()
    
    return best_h, best_val


# ---------------------------------------------------------------------------
# Incremental single-flip autocorrelation update
# ---------------------------------------------------------------------------

def _flip_ac_delta(h, i):
    """
    Compute the change in the autocorrelation array when flipping h[i].
    Returns (delta_ac, affected_lags) where delta_ac is a dict mapping
    lag -> delta, and affected_lags is the set of lags that changed.
    """
    # Flipping h[i] -> -h[i]
    # ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
    # When we flip h[i], for each lag k:
    #   If 0 <= i <= N-1-k: h[i] appears as h[j] with j=i, so term h[i]*h[i+k] changes sign
    #   If 0 <= i-k <= N-1-k: h[i] appears as h[j+k] with j=i-k, so term h[i-k]*h[i] changes sign
    # The change in ac[k] is:
    #   -2 * (contribution from h[i])
    # where contribution is h[i]*h[i+k] if i+k < N, plus h[i-k]*h[i] if i-k >= 0
    
    # For lag k >= 1:
    #   h[i] contributes to ac[k] as:
    #     - h[i]*h[i+k] if i+k < N (i is the first index)
    #     - h[i-k]*h[i] if i-k >= 0 (i is the second index)
    # After flip, these become -h[i]*h[i+k] and -h[i-k]*h[i]
    # So delta_ac[k] = -2 * (h[i]*h[i+k] if i+k<N else 0) - 2 * (h[i-k]*h[i] if i-k>=0 else 0)
    
    # We'll compute delta for all k in 1..N-1
    # But only k where i+k < N or i-k >= 0 are affected
    # k ranges from max(1, i-(N-1)) to min(N-1, N-1-i) for the first term
    # and k ranges from max(1, N-i) to min(N-1, i) for the second term
    # Actually let's just compute for all k=1..N-1 but only non-zero ones
    
    delta = np.zeros(N)
    # First term: h[i]*h[i+k] for k=1..N-1-i
    if i + 1 < N:
        k_range = np.arange(1, N - i)
        delta[k_range] -= 2.0 * h[i] * h[i + k_range]
    # Second term: h[i-k]*h[i] for k=1..i
    if i >= 1:
        k_range2 = np.arange(1, i + 1)
        delta[k_range2] -= 2.0 * h[i - k_range2] * h[i]
    
    return delta


def _eval_single_flip(h, ac, i):
    """
    Evaluate the peak sidelobe after flipping h[i], using incremental AC update.
    Returns (new_peak, new_ac) where new_ac is the updated AC array.
    """
    delta = _flip_ac_delta(h, i)
    new_ac = ac + delta
    new_peak = float(np.max(np.abs(new_ac[1:])))
    return new_peak, new_ac


def _apply_flip(h, ac, i):
    """
    Apply flip at position i to both h and ac.
    Returns updated h and ac.
    """
    delta = _flip_ac_delta(h, i)
    ac = ac + delta
    h[i] *= -1.0
    return h, ac


# ---------------------------------------------------------------------------
# Greedy polish: single-flip + adjacent-pair-flip (with incremental AC)
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start):
    h = h.copy()
    ac = _full_ac(h)
    cur = _peak_from_ac(ac)
    best = cur
    best_h = h.copy()
    best_ac = ac.copy()
    while time.time() - start < time_budget:
        # --- single-flip sweep ---
        best_i = -1
        best_c = cur
        for i in range(N):
            c, _ = _eval_single_flip(h, ac, i)
            if c < best_c:
                best_c = c
                best_i = i
        if best_i >= 0:
            h, ac = _apply_flip(h, ac, best_i)
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
                best_ac = ac.copy()
        # --- adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            # Flip both i and i+1
            # Compute delta for flipping i, then for flipping i+1 on the already-flipped h
            # But we need to be careful: after flipping i, h[i] is negated
            # Let's compute the combined delta directly
            # Original: h[i], h[i+1]
            # After flip: -h[i], -h[i+1]
            # The change in ac[k] from flipping both:
            # For each lag k, the terms involving h[i] or h[i+1] change
            # This is more complex; let's just do two sequential flips for correctness
            # but compute the combined effect
            delta1 = _flip_ac_delta(h, i)
            # After flipping i, h[i] is negated. Now flip i+1.
            # For the second flip, we need h with h[i] negated
            h_temp = h.copy()
            h_temp[i] *= -1.0
            delta2 = _flip_ac_delta(h_temp, i + 1)
            total_delta = delta1 + delta2
            new_ac = ac + total_delta
            c = float(np.max(np.abs(new_ac[1:])))
            if c < best_c2:
                best_c2 = c
                best_pair = i
        if best_pair >= 0:
            # Apply both flips
            h, ac = _apply_flip(h, ac, best_pair)
            h, ac = _apply_flip(h, ac, best_pair + 1)
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
                best_ac = ac.copy()
        # --- break if neither sweep improved ---
        if best_i < 0 and best_pair < 0:
            break
    return best_h, best


def _greedy_single_only(h, time_budget, start):
    """Tighter single-flip polish for the final pass (with incremental AC)."""
    h = h.copy()
    ac = _full_ac(h)
    cur = _peak_from_ac(ac)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            c, _ = _eval_single_flip(h, ac, i)
            if c < best_c:
                best_c = c
                best_i = i
        if best_i < 0:
            break
        h, ac = _apply_flip(h, ac, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def _fast_local_search(h, time_budget, start):
    """
    Fast local search using vectorized flip evaluation.
    Tries all single flips in parallel using matrix operations.
    """
    h = h.copy()
    ac = _full_ac(h)
    cur = _peak_from_ac(ac)
    best = cur
    best_h = h.copy()
    
    # Precompute the flip deltas for all positions at once
    # delta[i][k] = change in ac[k] when flipping h[i]
    # delta[i][k] = -2*h[i]*(h[i+k] if i+k<N else 0) - 2*h[i-k]*h[i] if i-k>=0 else 0
    
    while time.time() - start < time_budget:
        # Compute all single-flip peaks in parallel
        # For each i, new_ac = ac + delta_i
        # peak_i = max_k |ac[k] + delta_i[k]|
        
        # Vectorized computation
        # delta for flip at position i:
        # For lag k: delta[i,k] = -2*h[i]*h[i+k] (if i+k<N) - 2*h[i-k]*h[i] (if i-k>=0)
        
        # Build delta matrix: shape (N, N) where delta[i,k] is the change
        # This is O(N^2) memory but N=400 so 160K entries, fine
        
        # More efficient: compute the peak for each flip
        # new_ac[i,k] = ac[k] + delta[i,k]
        # peak[i] = max_k |new_ac[i,k]|
        
        # Let's compute delta as a matrix
        # delta[i,k] for i=0..N-1, k=1..N-1
        # = -2*h[i]*h[i+k] * 1[i+k<N] - 2*h[i-k]*h[i] * 1[i-k>=0]
        
        # Use broadcasting
        idx = np.arange(N)
        # First term: -2*h[i]*h[i+k] for valid i+k
        # h[i+k] where i+k < N
        # Create matrix M1[i,k] = h[i+k] if i+k<N else 0
        # M1[i,k] = h[i+k] for k < N-i
        
        # Simpler approach: just loop over i but vectorize the max
        best_i = -1
        best_c = cur
        
        # For speed, only evaluate flips that could improve
        # A flip at i can improve if the current peak lags are affected
        # For now, just do all flips but with early termination
        
        for i in range(N):
            # Quick check: if this flip doesn't affect the current peak lags, skip
            # The peak is at some lag k*. If flipping i doesn't change ac[k*], skip
            # But this is hard to check quickly, so just compute
            
            delta = _flip_ac_delta(h, i)
            new_ac = ac + delta
            c = float(np.max(np.abs(new_ac[1:])))
            if c < best_c:
                best_c = c
                best_i = i
        
        if best_i < 0:
            break
        h, ac = _apply_flip(h, ac, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    
    return best_h, best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 1: Continuous relaxation from random start (multiple seeds)
        rng = np.random.default_rng(42)
        for seed in range(3):
            if _remaining() < 30.0:
                break
            h_init = rng.choice([-1.0, 1.0], size=N)
            budget1 = min(25.0, _remaining() - 30.0)
            if budget1 > 5.0:
                h, c = _continuous_relax(h_init, budget1, time.time(), seed=seed)
                if c < global_best:
                    global_best = c
                    global_best_h = h

        # Phase 2: Greedy polish
        budget2 = min(20.0, _remaining() - 20.0)
        if budget2 > 2.0:
            h2, c2 = _greedy_polish(global_best_h, budget2, time.time())
            if c2 < global_best:
                global_best = c2
                global_best_h = h2

        # Phase 3: Another continuous relaxation with different seed
        budget3 = min(20.0, _remaining() - 15.0)
        if budget3 > 3.0:
            h3_init = global_best_h.copy()
            h3, c3 = _continuous_relax(h3_init, budget3, time.time(), seed=10)
            if c3 < global_best:
                global_best = c3
                global_best_h = h3

        # Phase 4: Fast local search
        budget4 = min(15.0, _remaining() - 10.0)
        if budget4 > 1.0:
            h4, c4 = _fast_local_search(global_best_h, budget4, time.time())
            if c4 < global_best:
                global_best = c4
                global_best_h = h4

        # Phase 5: Final greedy polish
        budget5 = min(15.0, _remaining() - 5.0)
        if budget5 > 1.0:
            h5, c5 = _greedy_single_only(global_best_h, budget5, time.time())
            if c5 < global_best:
                global_best = c5
                global_best_h = h5

        # Phase 6: One more polish if time remains
        budget6 = _remaining() - 2.0
        if budget6 > 0.5:
            h6, c6 = _greedy_polish(global_best_h, budget6, time.time())
            if c6 < global_best:
                global_best = c6
                global_best_h = h6

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
