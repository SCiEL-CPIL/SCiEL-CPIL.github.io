import time
import numpy as np
from scipy import signal

N = 400
START_TIME = time.time()
DEADLINE = START_TIME + 135.0  # safety margin before 150s


def _c3_prime_val(h):
    """
    Compute max_k |autocorr_k| / (sum h)^2 for signed step function h of length N.
    """
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if abs(total) < 1e-12:
        return np.inf
    denom = total * total
    ac = np.correlate(h, h, mode='full')
    return np.max(np.abs(ac)) / denom


def _fast_flip_eval(h, i, current_ac, current_total):
    """
    Evaluate the C3' value after flipping h[i], given current autocorrelation
    and total. Returns (new_val, new_ac, new_total) without full recompute.
    """
    h = np.asarray(h, dtype=float)
    flip_sign = -2.0 * h[i]  # change in h[i]
    
    # Change in autocorrelation: ac[k] += flip_sign * h[i+k] for valid k
    # ac is full mode: index j corresponds to lag (j - (N-1))
    # h[i] * h[i+k] contributes to ac at lag k, which is index (N-1) + k
    # When we flip h[i], the change in ac at lag k is flip_sign * h[i+k]
    # i+k must be in [0, N-1], so k in [-i, N-1-i]
    # ac index = (N-1) + k
    
    new_ac = current_ac.copy()
    new_total = current_total + flip_sign
    
    # For each valid k where i+k is in range:
    # k ranges from -i to N-1-i
    k_min = -i
    k_max = N - 1 - i
    # ac indices: (N-1) + k_min to (N-1) + k_max
    idx_min = (N - 1) + k_min
    idx_max = (N - 1) + k_max
    
    # h[i+k] for k in [k_min, k_max] is h[0 : N-i] if k_min=0... 
    # Actually h[i+k] for k from k_min to k_max:
    # i + k_min = 0, i + k_max = N-1
    # So it's h[0:N]
    # But wait: k_min = -i, so i + k_min = 0
    # k_max = N-1-i, so i + k_max = N-1
    # So h[i+k] for k in [k_min, k_max] = h[0:N]
    
    new_ac[idx_min:idx_max+1] += flip_sign * h[0:N]
    
    # Also, the autocorrelation is symmetric: ac[k] = ac[-k]
    # But np.correlate(h,h,'full') already handles this correctly.
    # Actually, let me reconsider. The autocorrelation R[k] = sum_j h[j]*h[j+k]
    # When h[i] changes by delta, R[k] changes by delta*h[i+k] for each k where i+k is valid.
    # But also R[-k] changes by delta*h[i-k]... 
    # Actually np.correlate(h, h, 'full') computes sum_j h[j]*conj(h[j+k]) for the correlation.
    # For real signals, R[k] = sum_j h[j]*h[j+k].
    # Flipping h[i] by delta: R[k] += delta * h[i+k] for k where i+k in [0,N-1]
    # AND R[k] += delta * h[i-k] for k where i-k in [0,N-1]? 
    # No. Let me think again.
    # R[k] = sum_j h[j] * h[j+k]
    # The term involving h[i] is when j=i: h[i]*h[i+k], and when j+k=i: h[j]*h[i] = h[i-k]*h[i]
    # So R[k] has two contributions from h[i]: h[i]*h[i+k] (when j=i) and h[i-k]*h[i] (when j=i-k)
    # So delta_R[k] = delta*h[i+k] + delta*h[i-k] (where indices are valid)
    
    # Let me redo this properly.
    # R[k] = sum_j h[j]*h[j+k]
    # Contribution of h[i] to R[k]:
    #   - h[i]*h[i+k] when j=i (requires i+k in [0,N-1])
    #   - h[i-k]*h[i] when j+k=i, i.e., j=i-k (requires i-k in [0,N-1])
    # So delta_R[k] = delta*h[i+k] + delta*h[i-k]
    
    # This is more complex. Let me just use the full recompute for correctness
    # but do it efficiently with FFT for the candidate evaluation phase.
    
    return None  # Not using this fast path, will use FFT-based approach


def _c3_prime_fft(h):
    """Compute C3' using FFT for speed."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if abs(total) < 1e-12:
        return np.inf
    denom = total * total
    # Use FFT for linear autocorrelation
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
    # For linear autocorrelation, we need the full version
    # Actually, np.correlate(h, h, 'full') gives 2N-1 values
    # The FFT approach: ac_full[k] for k=0..N-1 is the "positive" lags
    # Let me just use np.correlate for correctness
    ac = np.correlate(h, h, mode='full')
    return np.max(np.abs(ac)) / denom


def _local_search_single_flip(h, max_sweeps=50):
    """
    Local search: try flipping each single element, keep improvements.
    Uses incremental update for speed.
    """
    h = h.copy().astype(float)
    total = h.sum()
    ac = np.correlate(h, h, mode='full')
    current_val = np.max(np.abs(ac)) / (total * total)
    
    improved = True
    sweeps = 0
    while improved and sweeps < max_sweeps and time.time() < DEADLINE - 10:
        improved = False
        sweeps += 1
        
        for i in range(N):
            if time.time() > DEADLINE - 10:
                break
            
            # Try flipping h[i]
            trial = h.copy()
            trial[i] *= -1
            val = _c3_prime_val(trial)
            if val < current_val - 1e-12:
                h = trial
                current_val = val
                total = h.sum()
                ac = np.correlate(h, h, mode='full')
                improved = True
    
    return h, current_val


def _local_search_segment(h, seg_lengths=[2, 3, 4, 5, 7, 8], max_sweeps=30):
    """Local search: try flipping small segments."""
    h = h.copy().astype(float)
    current_val = _c3_prime_val(h)
    
    improved = True
    sweeps = 0
    while improved and sweeps < max_sweeps and time.time() < DEADLINE - 10:
        improved = False
        sweeps += 1
        
        for seg_len in seg_lengths:
            for start in range(0, N - seg_len + 1, max(1, seg_len // 2)):
                if time.time() > DEADLINE - 10:
                    break
                trial = h.copy()
                trial[start:start+seg_len] *= -1
                val = _c3_prime_val(trial)
                if val < current_val - 1e-12:
                    h = trial
                    current_val = val
                    improved = True
    
    return h, current_val


def _walsh_hadamard_sequence(n, index):
    """Generate the n-th Walsh-Hadamard sequence of length 2^k >= n."""
    k = 1
    while (1 << k) < n:
        k += 1
    size = 1 << k
    
    def hadamard(m):
        if m == 1:
            return np.array([[1.0]])
        H = hadamard(m // 2)
        return np.block([[H, H], [H, -H]])
    
    H = hadamard(size)
    row = H[index % size]
    return row[:n]


def _fourier_basis_signal(n, freq_index, phase=0.0):
    """Generate a sign(sin) signal at a specific Fourier frequency."""
    t = np.arange(n)
    return np.sign(np.sin(2 * np.pi * freq_index * t / n + phase))


def _random_zero_mean(n, seed):
    """Generate a random ±1 sequence with exactly zero mean (or near-zero)."""
    rng = np.random.RandomState(seed)
    h = rng.choice([-1.0, 1.0], size=n)
    total = h.sum()
    if total > 0:
        n_flip = int(total / 2)
        if n_flip > 0:
            plus_indices = np.where(h == 1.0)[0]
            flip_idx = rng.choice(plus_indices, size=min(n_flip, len(plus_indices)), replace=False)
            h[flip_idx] = -1.0
    elif total < 0:
        n_flip = int(-total / 2)
        if n_flip > 0:
            minus_indices = np.where(h == -1.0)[0]
            flip_idx = rng.choice(minus_indices, size=min(n_flip, len(minus_indices)), replace=False)
            h[flip_idx] = 1.0
    return h


def construct():
    global START_TIME, DEADLINE
    
    # Start with all-ones (the seed that local search improves well)
    best_h = np.ones(N)
    best_val = _c3_prime_val(best_h)
    
    # Generate diverse candidate patterns
    candidates = []
    
    # 1. Walsh-Hadamard sequences
    for idx in range(0, 32):
        candidates.append(_walsh_hadamard_sequence(N, idx))
    
    # 2. Fourier basis signals
    for freq in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        for phase in [0, np.pi/4, np.pi/2, np.pi/3]:
            candidates.append(_fourier_basis_signal(N, freq, phase))
    
    # 3. Random zero-mean sequences
    for seed in range(80):
        candidates.append(_random_zero_mean(N, seed))
    
    # 4. Alternating patterns
    for block_size in [2, 3, 4, 5, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100]:
        h = np.zeros(N)
        sign = 1.0
        for i in range(N):
            h[i] = sign
            if (i + 1) % block_size == 0:
                sign = -sign
        candidates.append(h)
    
    # 5. Chirp patterns
    t = np.arange(N)
    for f0 in [1, 2, 5]:
        for f1 in [5, 10, 20, 50]:
            if f1 != f0:
                freqs = f0 + (f1 - f0) * t / N
                phase = 2 * np.pi * np.cumsum(freqs) / N
                candidates.append(np.sign(np.sin(phase)))
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > DEADLINE:
            break
        try:
            val = _c3_prime_val(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
        except Exception:
            continue
    
    # Aggressive local search from all-ones (known to work well per knowledge)
    if time.time() < DEADLINE - 40:
        # Start from all-ones and do single-flip local search
        h_ones = np.ones(N)
        val_ones = _c3_prime_val(h_ones)
        
        # Single flip local search from all-ones
        h_refined, val_refined = _local_search_single_flip(h_ones, max_sweeps=40)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined
        
        # Also do segment flips from all-ones
        if time.time() < DEADLINE - 20:
            h_seg, val_seg = _local_search_segment(h_ones, seg_lengths=[2, 3, 4, 5, 7, 8], max_sweeps=20)
            if val_seg < best_val:
                best_val = val_seg
                best_h = h_seg
    
    # Local refinement from current best
    if time.time() < DEADLINE - 15:
        current_h = best_h.copy()
        current_val = best_val
        
        # Single flip refinement
        current_h, current_val = _local_search_single_flip(current_h, max_sweeps=30)
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
        
        # Segment refinement
        if time.time() < DEADLINE - 10:
            current_h, current_val = _local_search_segment(current_h, seg_lengths=[2, 3, 4, 5, 7, 8, 12], max_sweeps=15)
            if current_val < best_val:
                best_val = current_val
                best_h = current_h
    
    return best_h
