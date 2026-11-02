"""Low-peak-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound
below the all-ones baseline.

Design family (kept): simulated annealing + greedy polish, now with
  * an O(N) incremental autocorrelation update for single-flips (no FFT per trial),
  * a smooth epigraph / soft-peak (log-sum-exp) surrogate for exploration,
  * an ADMM / projected-gradient pass on the continuous relaxation h in [-1,1],
  * a high-precision greedy single-flip + adjacent-pair-flip polish,
  * an explicit 140 s deadline guard (10 s safety under the 150 s kill),
  * sine-based spectral initialization to exploit low-frequency structure.
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def _ac_full(h):
    """Full aperiodic autocorrelation ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k], k=0..N-1."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe(h):
    ac = _ac_full(h)
    return float(np.max(np.abs(ac[1:])))


def _soft_peak(ac, tau):
    """Smooth epigraph surrogate: log-sum-exp of ac[k]^2/tau over k>=1."""
    v = ac[1:] ** 2 / tau
    m = float(np.max(v))
    return m + float(np.log(np.sum(np.exp(v - m))))


def _flip_ac(ac, h, i):
    """Return the change to ac after flipping h[i] (in-place not needed; returns delta).

    delta[k] = -2*h[i]*g where g = h[i+k] if i+k<N else h[i-k] if i-k>=0 else 0.
    We return the new ac as a fresh array to keep logic simple but cheap.
    """
    hi = h[i]
    new_ac = ac.copy()
    # k ranges where i and i+k are both in range: k in [0, N-1-i]
    # contribution of (i, i+k): index k, value h[i]*h[i+k]
    k_hi = N - 1 - i
    for k in range(0, k_hi + 1):
        new_ac[k] += -2.0 * hi * h[i + k]
    # contribution of (i-k, i): index k, value h[i-k]*h[i], for k in [1, i]
    k_lo = i
    for k in range(1, k_lo + 1):
        new_ac[k] += -2.0 * hi * h[i - k]
    return new_ac


def _flip_ac_fast(ac, h, i):
    """Vectorised O(N) update of ac after flipping h[i]."""
    new_ac = ac.copy()
    hi = h[i]
    k_hi = N - 1 - i
    if k_hi >= 0:
        new_ac[0:k_hi + 1] += -2.0 * hi * h[i:i + k_hi + 1]
    k_lo = i
    if k_lo >= 1:
        new_ac[1:k_lo + 1] += -2.0 * hi * h[i - k_lo:i]
    return new_ac


# ---------------------------------------------------------------------------
# Continuous relaxation: projected-gradient / ADMM on soft peak
# ---------------------------------------------------------------------------

def _soft_peak_grad(ac, tau):
    """Gradient of soft peak f = log sum e^{ac[k]^2/tau} wrt ac (k>=1)."""
    v = ac[1:] ** 2 / tau
    m = float(np.max(v))
    e = np.exp(v - m)
    p = e / float(np.sum(e))          # softmax weights
    # d f / d ac[k] = 2*ac[k]/tau * p[k]
    g = np.zeros(N)
    g[1:] = (2.0 / tau) * ac[1:] * p
    return g


def _continuous_relax(h0, time_budget, start):
    """Projected-gradient on the continuous relaxation h in [-1,1].

    Objective: soft peak of the autocorrelation (smooth epigraph surrogate).
    Gradient wrt h_i from the autocorrelation gradient wrt ac.
    Returns a +-1 sequence (rounded) and its peak.
    """
    rng = np.random.default_rng(7)
    h = h0.astype(np.float64).copy()
    # start from a rounded +-1 but allow continuous moves
    tau = 8.0
    step = 0.05
    ac = _ac_full(h)
    f = _soft_peak(ac, tau)
    best_f = f
    best_h = h.copy()
    while time.time() - start < time_budget:
        g_ac = _soft_peak_grad(ac, tau)
        # Vectorized gradient computation:
        # grad_h[i] = sum_{k=1}^{N-1} g_ac[k] * (h[i+k] if i+k<N else 0 + h[i-k] if i-k>=0 else 0)
        # This can be written as: grad_h = sum_k g_ac[k] * (shift_right(h, k) + shift_left(h, k))
        # where shift_right(h, k)[i] = h[i+k] if i+k<N else 0
        # and shift_left(h, k)[i] = h[i-k] if i-k>=0 else 0
        
        # Use FFT-based approach or direct vectorization
        # Direct vectorization: for each k, add g_ac[k] * (h[k:] padded + h[:-k] padded)
        grad_h = np.zeros(N)
        # Vectorize the loop using numpy broadcasting where possible
        # For k from 1 to N-1:
        # grad_h[0:N-k] += g_ac[k] * h[k:N]
        # grad_h[k:N] += g_ac[k] * h[0:N-k]
        
        # We can do this more efficiently by noting the structure
        # Let's use a different approach: compute via correlation-like operation
        # Actually, let's just vectorize the inner loop better
        
        # Build the gradient using the fact that it's a sum of shifted versions
        # For efficiency, we can use the following trick:
        # grad_h[i] = sum_{j: |j-i| >= 1} g_ac[|j-i|] * h[j]
        # This is like a weighted sum of h with weights depending on distance
        
        # Simple vectorized version:
        for k in range(1, N):
            gk = g_ac[k]
            if abs(gk) > 1e-10:  # Skip negligible contributions
                grad_h[0:N - k] += gk * h[k:N]
                grad_h[k:N] += gk * h[0:N - k]
        
        # Adaptive step size based on gradient magnitude
        grad_norm = np.linalg.norm(grad_h)
        if grad_norm > 1e-8:
            step_adaptive = step * (1.0 / (1.0 + grad_norm * 0.1))
        else:
            step_adaptive = step
        
        # projected gradient step
        h = h - step_adaptive * grad_h
        h = np.clip(h, -1.0, 1.0)
        ac = _ac_full(h)
        nf = _soft_peak(ac, tau)
        if nf < best_f:
            best_f = nf
            best_h = h.copy()
        # anneal tau down, occasional restart of step
        tau = max(1.0, tau * 0.99)
        step = 0.05 + 0.02 * (1.0 - min(1.0, tau / 8.0))
    # round to +-1
    hr = np.where(best_h >= 0.0, 1.0, -1.0)
    return hr, _peak_sidelobe(hr)


# ---------------------------------------------------------------------------
# Annealing with variable-length block flips (uses soft surrogate to accept)
# ---------------------------------------------------------------------------

def _sine_init(freq, phase, seed):
    """Generate a +-1 sequence from zero crossings of a sine wave with noise.
    
    This creates smooth transitions between positive and negative blocks,
    which can reduce high-frequency energy in the autocorrelation.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi * freq, N)
    signal = np.sin(x + phase)
    # Add small noise to break perfect periodicity
    noise = rng.normal(0, 0.1, size=N)
    h = np.sign(signal + noise)
    h[h == 0] = 1.0
    return h

def _vdc_init(seed):
    """Generate a +-1 sequence based on the van der Corput sequence.
    
    The van der Corput sequence is a low-discrepancy sequence that provides
    near-uniform distribution in [0,1]. Mapping this to +-1 creates a sequence
    with good spectral flatness properties, which helps in reducing autocorrelation peaks.
    """
    # Generate van der Corput sequence values
    vdc = np.zeros(N)
    for i in range(N):
        # van der Corput: reverse binary representation
        # Simpler: use the property that vdc(i) = sum_{k=0}^{infty} bit_k(i) / 2^{k+1}
        # We can compute this efficiently or just use a standard implementation
        # For N=400, a simple loop is fine, or use numpy bit operations
        pass
    
    # Efficient VDC generation
    # vdc(i) = 0.b_0 b_1 ... in base 2, where b_k is the k-th bit of i (LSB first)
    # Actually, standard VDC is 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8...
    # vdc(i) = sum_{j=0}^{infty} floor(i / 2^j) % 2 / 2^{j+1}
    
    vdc_vals = np.zeros(N)
    for i in range(N):
        val = 0.0
        div = 1
        n = i
        while n > 0:
            div *= 2
            if n % 2 == 1:
                val += 1.0 / div
            n //= 2
        vdc_vals[i] = val
        
    # Map [0, 1) to {-1, 1}
    # Threshold at 0.5
    h = np.where(vdc_vals < 0.5, -1.0, 1.0)
    
    # Add slight random perturbation to break symmetries if needed, 
    # but VDC is deterministic. We can use seed to shift the sequence.
    shift = seed % N
    h = np.roll(h, shift)
    
    return h


def _anneal(seed, time_budget, init_h=None):
    rng = np.random.default_rng(seed)
    if init_h is not None:
        h = init_h.copy()
    else:
        # Try multiple initialization strategies
        strategy = rng.integers(0, 3)
        if strategy == 0:
            h = _sine_init(rng.uniform(2, 8), rng.uniform(0, 2 * np.pi), seed)
        elif strategy == 1:
            h = _sine_init(rng.uniform(5, 15), rng.uniform(0, 2 * np.pi), seed)
        else:
            h = rng.choice([-1.0, 1.0], size=N)
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-7
    tau = 8.0
    start = time.time()
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        # Bias toward single-flip moves (faster) especially early on
        if progress < 0.7 and rng.random() < 0.6:
            L = 1
        else:
            max_len = max(1, int(24.0 * (1.0 - 0.5 * progress)))
            L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        
        # Fast path for single flips
        if L == 1:
            ac_new = _flip_ac_fast(ac, h, i)
            new = float(np.max(np.abs(ac_new[1:])))
            # Hybrid acceptance: improve peak OR improve soft-peak
            soft_diff = _soft_peak(ac_new, tau) - _soft_peak(ac, tau)
            if new < cur:
                accept = True
            elif soft_diff < 0:
                accept = True
            else:
                # Accept worsening moves with probability based on peak difference
                peak_diff = new - cur
                accept = rng.random() < np.exp(-peak_diff / max(T, 1e-9))
            
            if accept:
                h[i] *= -1.0
                ac = ac_new
                cur = new
                if cur < best:
                    best = cur
                    best_h = h.copy()
            # else: don't flip, ac stays the same
        else:
            # Multi-flip: use FFT
            h[i:i + L] *= -1.0
            ac_new = _ac_full(h)
            new = _peak_sidelobe(h)
            soft_diff = _soft_peak(ac_new, tau) - _soft_peak(ac, tau)
            if new < cur or soft_diff < 0 or rng.random() < np.exp(-soft_diff / max(T, 1e-9)):
                cur = new
                ac = ac_new
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                h[i:i + L] *= -1.0
        
        T = max(T_min, T * 0.99995)
        tau = max(1.0, tau * 0.9995)
    return best_h, best


# ---------------------------------------------------------------------------
# Greedy polish: single-flip (O(N) update) + adjacent-pair-flip
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start):
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    # Optimization: Pre-compute indices of current peaks to check if they are reduced
    # But for simplicity and robustness, we stick to full max but optimize the loop.
    # We can skip the pair-flip if it's too slow, as single-flip is often sufficient.
    # Let's try to make the single-flip sweep faster by checking only if the new peak
    # is less than the current peak.
    
    while time.time() - start < time_budget:
        # --- single-flip sweep (fast O(N) update) ---
        best_i = -1
        best_c = cur
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            # Quick check: if the max absolute value is not less than cur, skip detailed check?
            # No, we need the exact max.
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
                best_c = c
                best_i = i
        if best_i >= 0:
            h[best_i] *= -1.0
            ac = _flip_ac_fast(ac, h, best_i)
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # No single flip improved it. Try pair flips only if time remains and budget allows.
            # Pair flips are O(N log N) per trial, so limit them.
            if time_budget - (time.time() - start) > 1.0:
                best_pair = -1
                best_c2 = cur
                # Limit pair flip search to a subset or break early if no improvement found quickly
                for i in range(N - 1):
                    if time.time() - start > time_budget - 0.5:
                        break
                    h[i] *= -1.0
                    h[i + 1] *= -1.0
                    ac2 = _ac_full(h)
                    c = float(np.max(np.abs(ac2[1:])))
                    h[i] *= -1.0
                    h[i + 1] *= -1.0
                    if c < best_c2:
                        best_c2 = c
                        best_pair = i
                if best_pair >= 0:
                    h[best_pair] *= -1.0
                    h[best_pair + 1] *= -1.0
                    ac = _ac_full(h)
                    cur = best_c2
                    if cur < best:
                        best = cur
                        best_h = h.copy()
            else:
                break
        if best_i < 0 and (best_pair < 0 if 'best_pair' in locals() else False):
             # If no single flip worked and we didn't even try pair flips or they failed
             if best_i < 0 and not (time_budget - (time.time() - start) > 1.0):
                 break
             # If we tried pair flips and none worked, we are at a local minimum
             # We can break or continue if we want to try more complex moves, but let's break to save time
             if best_i < 0 and (best_pair < 0 if 'best_pair' in locals() else True):
                 break
    return best_h, best


def _greedy_single_only(h, time_budget, start):
    """Tight single-flip polish with the O(N) update."""
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    # Precompute which indices are at the peak to check if they're reduced
    peak_indices = np.where(np.abs(ac[1:]) >= cur - 1e-10)[0] + 1
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        # Only check flips that might affect the peak
        # A flip at position i affects ac[k] for k where |i-j|=k for some j
        # The peak is at peak_indices, so we need to check if flipping i reduces any of those
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            # Quick check: if the max is not less than cur, skip
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c - 1e-12:
                best_c = c
                best_i = i
        if best_i < 0:
            break
        h[best_i] *= -1.0
        ac = _flip_ac_fast(ac, h, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
        # Update peak indices
        peak_indices = np.where(np.abs(ac[1:]) >= cur - 1e-10)[0] + 1
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

        # Phase 1: continuous relaxation (ADMM / projected gradient) to seed
        # a good +-1 starting point, then a quick polish.
        budget = min(18.0, _remaining() - 25.0)
        if budget > 3.0:
            h, c = _continuous_relax(global_best_h, budget, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(6.0, _remaining() - 25.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: long initial anneal from spectral initialization.
        budget = min(24.0, _remaining() - 22.0)
        if budget > 2.0:
            # Try sine-based initialization
            h_sine = _sine_init(5.0, np.pi / 4, 12345)
            h, c = _anneal(12345, budget, init_h=h_sine)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(5.0, _remaining() - 22.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 3: multiple shorter restarts for diversity with spectral init.
        restart_id = 1
        while _remaining() > 18.0:
            sub = min(12.0, _remaining() - 18.0)
            if sub < 1.5:
                break
            # Alternate between different spectral initializations, including VDC
            if restart_id % 4 == 1:
                h_init = _sine_init(3.0, 0.0, 100000 + restart_id)
            elif restart_id % 4 == 2:
                h_init = _sine_init(7.0, np.pi / 2, 100000 + restart_id)
            elif restart_id % 4 == 3:
                h_init = _vdc_init(100000 + restart_id)
            else:
                h_init = None # Random
            h, c = _anneal(100000 + restart_id, sub, init_h=h_init)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(3.0, _remaining() - 18.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 4: final high-precision single-flip polish.
        remaining = _remaining() - 3.0
        if remaining > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 5: one more polish pass if time remains.
        remaining2 = _remaining() - 1.0
        if remaining2 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 6: Final fine-tuning with very small steps
        remaining3 = _remaining() - 0.5
        if remaining3 > 0.3:
            # Try a few more single flips with tighter tolerance
            h_final = global_best_h.copy()
            ac_final = _ac_full(h_final)
            cur_final = _peak_sidelobe(h_final)
            for _ in range(100):  # Limited iterations
                if time.time() - start > hard_deadline - 0.3:
                    break
                improved = False
                for i in range(N):
                    ac2 = _flip_ac_fast(ac_final, h_final, i)
                    c2 = float(np.max(np.abs(ac2[1:])))
                    if c2 < cur_final - 1e-12:
                        h_final[i] *= -1.0
                        ac_final = ac2
                        cur_final = c2
                        improved = True
                        break
                if not improved:
                    break
            if cur_final < global_best:
                global_best = cur_final
                global_best_h = h_final

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
