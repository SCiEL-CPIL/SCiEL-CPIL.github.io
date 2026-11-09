"""Erdos minimum overlap: structured combinatorial search."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 140.0  # safety margin under the 150 s kill


def _overlap_profile_fft(h):
    """Return overlap(k) = sum_i h[i] * (1 - h[(i-k) mod N]) for all cyclic lags k.
    Uses FFT-based circular cross-correlation for O(N log N) computation."""
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    OMH = np.fft.fft(one_minus_h)
    r = np.fft.ifft(H * np.conj(OMH)).real
    return r


def _score(h):
    """The quantity the evaluator minimizes: max_k overlap(h, shifted (1-h))."""
    h = np.asarray(h, dtype=float)
    return float(np.max(_overlap_profile_fft(h)))


def _fix_sum(h):
    """Adjust h so its sum is exactly N/2, keeping values in [0,1] if possible."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    diff = N / 2.0 - h.sum()
    if abs(diff) < 1e-12:
        return h
    # Distribute the difference; check it won't push out of [0,1]
    step = diff / N
    h2 = h + step
    if h2.min() >= -1e-12 and h2.max() <= 1.0 + 1e-12:
        h2 = np.clip(h2, 0.0, 1.0)
        # Tiny residual fix
        residual = N / 2.0 - h2.sum()
        if abs(residual) < 1e-9:
            h2 += residual / N
        return h2
    # Fallback: just add the step and clip (may not be exact, but rare)
    h2 = np.clip(h + step, 0.0, 1.0)
    return h2


def _construct_balanced_pattern(period, weights):
    """
    Construct a sequence by repeating a base pattern of length 'period'.
    weights is a list of values for the base pattern.
    We repeat it floor(N/period) times and handle the remainder by
    taking the first (N % period) elements of the base pattern.
    """
    if period <= 0:
        return None
    base = np.array(weights, dtype=float)
    # Ensure base sum is close to period/2 for balancedness, but we will fix_sum anyway
    # Repeat base to fill N
    reps = N // period
    rem = N % period
    if rem == 0:
        h = np.tile(base, reps)
    else:
        h = np.concatenate([np.tile(base, reps), base[:rem]])
    return h


def construct():
    global _T0
    # Baseline: constant 0.5
    best = np.full(N, 0.5)
    best_score = _score(best)  # should be exactly 25.0

    def deadline_ok():
        return (time.time() - _T0) < _DEADLINE

    candidates = []

    # Strategy: Search over periodic patterns with small periods.
    # For each period p, try various "balanced" weight distributions.
    # A key insight: if we can find a pattern where the overlap profile is flatter, we win.
    # For binary-like patterns (0/1), the overlap is related to difference sets.
    # Let's explore periods that divide N or are close to it.
    
    periods_to_try = [2, 4, 5, 10, 20, 25, 50]
    
    for p in periods_to_try:
        if not deadline_ok():
            break
            
        # Generate candidate base patterns of length p
        # We want sum(base) ~= p/2
        
        # 1. Alternating patterns
        if p % 2 == 0:
            pat = np.array([0.5, 0.5] * (p // 2))
            candidates.append(_construct_balanced_pattern(p, pat))
            
            # 0/1 alternating
            pat01 = np.array([1.0, 0.0] * (p // 2))
            candidates.append(_construct_balanced_pattern(p, pat01))
            
            # 0.25 / 0.75 alternating
            pat23 = np.array([0.25, 0.75] * (p // 2))
            candidates.append(_construct_balanced_pattern(p, pat23))
            
            # Block patterns: 1s then 0s
            for frac in [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]:
                n_ones = int(round(p * frac))
                pat_block = np.zeros(p)
                pat_block[:n_ones] = 1.0
                candidates.append(_construct_balanced_pattern(p, pat_block))
                
        # 2. Sinusoidal patterns on the base period
        for freq in [1, 2, 3]:
            for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                i = np.arange(p)
                pat_sin = 0.5 + 0.4 * np.sin(2 * np.pi * freq * i / p + phase)
                candidates.append(_construct_balanced_pattern(p, pat_sin))
                
        # 3. Random patterns on the base period (fixed seed for reproducibility within run)
        rng = np.random.default_rng(p)
        for _ in range(20):
            pat_rand = rng.uniform(0, 1, size=p)
            candidates.append(_construct_balanced_pattern(p, pat_rand))
            
        # 4. "Goldbach-like" or difference set inspired
        # For p=5, the (5,2,1) difference set is {0,1,2,3,4} no, that's all.
        # (5,2,1) doesn't exist. (7,3,1) does.
        # Let's try specific known low-discrepancy sequences if p is small.
        # For p=7: 1,0,1,0,1,0,0 (sum 3)
        if p == 7:
            candidates.append(_construct_balanced_pattern(7, [1,0,1,0,1,0,0]))
            candidates.append(_construct_balanced_pattern(7, [1,1,0,1,0,0,0]))
            
    # Also try non-periodic structured sequences
    # 1. Thue-Morse like (balanced)
    # 2. Van der Corput / Low discrepancy
    # Let's just generate a few more random global patterns to be safe, but focus is periodic
    rng_global = np.random.default_rng(99)
    for _ in range(50):
        if not deadline_ok():
            break
        arr = rng_global.uniform(0.0, 1.0, size=N)
        candidates.append(arr)
        
    # Evaluate all candidates
    for arr in candidates:
        if not deadline_ok():
            break
        if arr is None:
            continue
        arr = _fix_sum(arr)
        sc = _score(arr)
        if sc < best_score - 1e-10:
            best = arr.copy()
            best_score = sc

    # Final deadline guard
    if time.time() - _T0 > _DEADLINE - 2.0:
        return best
    return best
# EVOLVE-BLOCK-END
