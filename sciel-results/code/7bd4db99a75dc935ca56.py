"""Low-peak-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound
below the all-ones baseline.

Design: simulated annealing with O(N) incremental autocorrelation updates,
diverse spectral initializations, and multi-scale greedy polish.
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


def _peak_from_ac(ac):
    """Peak sidelobe from precomputed ac array."""
    return float(np.max(np.abs(ac[1:])))


def _soft_peak(ac, tau):
    """Smooth epigraph surrogate: log-sum-exp of ac[k]^2/tau over k>=1."""
    v = ac[1:] ** 2 / tau
    m = float(np.max(v))
    return m + float(np.log(np.sum(np.exp(v - m))))


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


def _block_flip_ac(ac, h, i, L):
    """O(N*L) update of ac after flipping h[i:i+L] (block of length L).
    
    For a block flip, each element in the block is flipped. The change to ac[k]
    is the sum of individual flip contributions, but we can do it more efficiently
    by noting that flipping a block changes the contribution of each pair where
    exactly one element is in the block.
    """
    new_ac = ac.copy()
    # For each position p in the block [i, i+L), flip h[p]
    # Contribution to ac[k] from flipping h[p]:
    #   -2*h[p]*h[p+k] for k in [0, N-1-p]  (pairs (p, p+k))
    #   -2*h[p]*h[p-k] for k in [1, p]      (pairs (p-k, p))
    # But we need to be careful: when flipping multiple elements, the interactions
    # between flipped elements cancel out (both signs change, product unchanged).
    # So the total delta is just the sum of individual flip deltas computed with
    # the ORIGINAL h values.
    for p in range(i, i + L):
        hp = h[p]
        k_hi = N - 1 - p
        if k_hi >= 0:
            new_ac[0:k_hi + 1] += -2.0 * hp * h[p:p + k_hi + 1]
        k_lo = p
        if k_lo >= 1:
            new_ac[1:k_lo + 1] += -2.0 * hp * h[p - k_lo:p]
    return new_ac


def _block_flip_ac_fast(ac, h, i, L):
    """Vectorized O(N*L) block flip using FFT-based approach for speed.
    
    Key insight: flipping block [i, i+L) means h_new = h - 2*(block_mask)*h
    where block_mask is 1 on [i,i+L) and 0 elsewhere.
    The autocorrelation change can be computed via convolution.
    """
    if L == 1:
        return _flip_ac_fast(ac, h, i)
    
    # Create the flip mask
    flip_mask = np.zeros(N)
    flip_mask[i:i+L] = 1.0
    
    # delta_h = -2 * flip_mask * h (the change in h)
    delta_h = -2.0 * flip_mask * h
    
    # The change in autocorrelation:
    # ac_new[k] = ac[k] + sum_j delta_h[j]*h[j+k] + sum_j h[j]*delta_h[j+k] + sum_j delta_h[j]*delta_h[j+k]
    # But since delta_h[j]*delta_h[j+k] = 4*flip_mask[j]*h[j]*flip_mask[j+k]*h[j+k],
    # and we want the EXACT new ac, let's use:
    # ac_new = ac(h + delta_h) = ac(h) + 2*cross(h, delta_h) + ac(delta_h)
    
    # Compute cross-correlation h * delta_h via FFT
    H = np.fft.fft(h, _FFT_SIZE)
    D = np.fft.fft(delta_h, _FFT_SIZE)
    cross = np.fft.ifft(H * np.conj(D)).real[:N]  # sum_j h[j] * delta_h[j+k]
    cross_fwd = np.fft.ifft(D * np.conj(H)).real[:N]  # sum_j delta_h[j] * h[j+k]
    
    # ac(delta_h) - autocorrelation of delta_h
    D_ac = np.fft.ifft(D * np.conj(D)).real[:N]
    
    # ac_new[k] = ac[k] + cross_fwd[k] + cross[k] + D_ac[k]
    # Wait, let me be more careful:
    # ac_new[k] = sum_j (h[j]+delta_h[j]) * (h[j+k]+delta_h[j+k])
    #            = sum_j h[j]*h[j+k] + sum_j h[j]*delta_h[j+k] + sum_j delta_h[j]*h[j+k] + sum_j delta_h[j]*delta_h[j+k]
    #            = ac[k] + cross_fwd[k] + cross[k] + D_ac[k]
    
    new_ac = ac + cross_fwd + cross + D_ac
    return new_ac


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def _sine_init(freq, phase, seed):
    """Generate a +-1 sequence from zero crossings of a sine wave with noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi * freq, N)
    signal = np.sin(x + phase)
    noise = rng.normal(0, 0.1, size=N)
    h = np.sign(signal + noise)
    h[h == 0] = 1.0
    return h


def _msequence_init(seed):
    """Generate a +-1 sequence from a maximal-length LFSR (m-sequence).
    
    M-sequences have excellent autocorrelation properties (periodic),
    which gives a good starting point for aperiodic optimization.
    """
    rng = np.random.default_rng(seed)
    # Use a 9-bit LFSR (period 511, we take 400)
    # Tap positions for 9-bit m-sequence: x^9 + x^4 + 1
    state = rng.integers(1, 512)
    h = np.zeros(N)
    for i in range(N):
        h[i] = 1.0 if (state & 1) == 0 else -1.0
        # LFSR step: new bit = bit[8] XOR bit[3]
        new_bit = ((state >> 8) ^ (state >> 3)) & 1
        state = ((state << 1) | new_bit) & 0x1FF
    return h


def _vdc_init(seed):
    """Generate a +-1 sequence based on the van der Corput sequence."""
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
    h = np.where(vdc_vals < 0.5, -1.0, 1.0)
    shift = seed % N
    h = np.roll(h, shift)
    return h


def _random_init(seed):
    """Random +-1 initialization."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=N)


# ---------------------------------------------------------------------------
# Annealing with O(N) block flip updates
# ---------------------------------------------------------------------------

def _anneal(seed, time_budget, init_h=None):
    """Simulated annealing with O(N) incremental autocorrelation updates.
    
    Uses block flips of variable length with soft-surrogate acceptance.
    Optimized for speed: uses FFT-based block flip for L>1.
    """
    rng = np.random.default_rng(seed)
    if init_h is not None:
        h = init_h.copy()
    else:
        strategy = rng.integers(0, 5)
        if strategy == 0:
            h = _sine_init(rng.uniform(2, 8), rng.uniform(0, 2 * np.pi), seed)
        elif strategy == 1:
            h = _sine_init(rng.uniform(5, 15), rng.uniform(0, 2 * np.pi), seed)
        elif strategy == 2:
            h = _msequence_init(seed)
        elif strategy == 3:
            h = _vdc_init(seed)
        else:
            h = _random_init(seed)
    
    ac = _ac_full(h)
    cur = _peak_from_ac(ac)
    best_h = h.copy()
    best = cur
    T = 8.0
    T_min = 1e-7
    tau = 16.0
    start = time.time()
    cur_soft = _soft_peak(ac, tau)
    iter_count = 0
    
    while time.time() - start < time_budget:
        elapsed = time.time() - start
        progress = min(1.0, elapsed / max(time_budget, 1e-9))
        
        # Annealing schedule: block length shrinks over time
        # Early: larger blocks for exploration; Late: single flips for exploitation
        if progress < 0.3:
            max_len = 16
        elif progress < 0.7:
            max_len = 8
        else:
            max_len = 2
        
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        
        # Apply block flip
        h[i:i + L] *= -1.0
        ac_new = _block_flip_ac_fast(ac, h, i, L)
        new = _peak_from_ac(ac_new)
        
        # Accept using soft-surrogate
        new_soft = _soft_peak(ac_new, tau)
        d = new_soft - cur_soft
        if new <= cur or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            ac = ac_new
            cur_soft = new_soft
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Undo
            h[i:i + L] *= -1.0
        
        T = max(T_min, T * 0.99995)
        tau = max(2.0, tau * 0.9999)
        
        # Periodically refresh soft peak with current tau
        if iter_count % 500 == 499:
            cur_soft = _soft_peak(ac, tau)
        
        iter_count += 1
    
    return best_h, best


# ---------------------------------------------------------------------------
# Greedy polish: single-flip + block-flip
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start):
    """Greedy polish with single-flip and small block-flip moves."""
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_from_ac(ac)
    best = cur
    best_h = h.copy()
    rng = np.random.default_rng(int(start * 1e6) % 2**31)
    
    while time.time() - start < time_budget:
        # --- single-flip sweep ---
        best_i = -1
        best_c = cur
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
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
            # No single flip improved. Try block flips of length 2-4.
            improved = False
            for L in [2, 3, 4, 6]:
                if time.time() - start > time_budget - 0.5:
                    break
                best_pos = -1
                best_c2 = cur
                # Sample positions for block flips
                n_starts = N - L
                n_sample = min(300, n_starts)
                sample_idx = rng.choice(n_starts, size=n_sample, replace=False)
                for i in sample_idx:
                    if time.time() - start > time_budget - 0.5:
                        break
                    ac2 = _block_flip_ac_fast(ac, h, int(i), L)
                    c = float(np.max(np.abs(ac2[1:])))
                    if c < best_c2:
                        best_c2 = c
                        best_pos = int(i)
                if best_pos >= 0:
                    h[best_pos:best_pos + L] *= -1.0
                    ac = _block_flip_ac_fast(ac, h, best_pos, L)
                    cur = best_c2
                    if cur < best:
                        best = cur
                        best_h = h.copy()
                    improved = True
            if not improved:
                break
    return best_h, best


def _greedy_single_only(h, time_budget, start):
    """Tight single-flip polish with O(N) update."""
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_from_ac(ac)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
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
    return best_h, best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    try:
        hard_deadline = 145.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 1: Initial anneal from m-sequence (excellent periodic autocorrelation)
        budget = min(30.0, _remaining() - 25.0)
        if budget > 3.0:
            h_mseq = _msequence_init(42)
            h, c = _anneal(42, budget, init_h=h_mseq)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(5.0, _remaining() - 25.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: Anneal from sine initialization
        budget = min(25.0, _remaining() - 20.0)
        if budget > 2.0:
            h_sine = _sine_init(5.0, np.pi / 4, 12345)
            h, c = _anneal(12345, budget, init_h=h_sine)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(4.0, _remaining() - 20.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 3: Multiple shorter restarts for diversity
        restart_id = 1
        while _remaining() > 15.0:
            sub = min(15.0, _remaining() - 15.0)
            if sub < 2.0:
                break
            
            # Cycle through diverse initializations
            if restart_id % 5 == 1:
                h_init = _msequence_init(100000 + restart_id)
            elif restart_id % 5 == 2:
                h_init = _sine_init(3.0, 0.0, 100000 + restart_id)
            elif restart_id % 5 == 3:
                h_init = _sine_init(7.0, np.pi / 2, 100000 + restart_id)
            elif restart_id % 5 == 4:
                h_init = _vdc_init(100000 + restart_id)
            else:
                h_init = None  # Random
            
            h, c = _anneal(100000 + restart_id, sub, init_h=h_init)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(3.0, _remaining() - 15.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 4: Final high-precision single-flip polish.
        remaining = _remaining() - 2.0
        if remaining > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 5: One more full polish pass if time remains.
        remaining2 = _remaining() - 0.5
        if remaining2 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
