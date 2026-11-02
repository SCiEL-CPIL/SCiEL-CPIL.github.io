"""Third autocorrelation inequality — model-and-optimize bootstrap."""
import numpy as np
import time

N = 400
DEADLINE = 110.0  # seconds, safety margin below 150s evaluator limit


def _compute_autocorr(h):
    """
    Compute the third-order autocorrelation (autoconvolution of f with f*f).
    For a discrete signal h of length N, C3[k] = sum_i h[i] * h[i+k] * h[i+2k]
    over valid i for each lag k.
    
    The "upper bound on C3" is the maximum absolute value of this triple product sum.
    """
    # Use FFT-based convolution for efficiency
    # f * f = conv(h, h), then C3 = conv(f*f, h)
    h_float = h.astype(np.float64)
    # Zero-pad to avoid circular convolution
    size = 2 * N + 1
    h_fft = np.fft.fft(h_float, n=size)
    # autoconv = conv(h, h)
    autoconv_fft = h_fft * h_fft
    # C3 = conv(autoconv, h)
    c3_fft = autoconv_fft * h_fft
    c3_full = np.fft.ifft(c3_fft).real
    # The linear convolution result of length 3N-1; c3_full[:3*N-1] is valid
    # But we only care about lags k=0,1,...,N-1 where all three h[i], h[i+k], h[i+2k] are in range
    # For lag k, the sum is over i=0 to N-1-2k, so valid for k = 0 to (N-1)//2
    # Take the first N elements as a conservative set of lags
    c3 = c3_full[:N]
    return c3


def _score(h):
    """The evaluator measures the raw value: max |C3| (lower is better)."""
    c3 = _compute_autocorr(h)
    return np.max(np.abs(c3))


def _make_sine_pattern(freq, phase, amp=1.0):
    """Generate a signed step function from a sine wave."""
    x = np.linspace(0, 1, N, endpoint=False)
    raw = amp * np.sin(2 * np.pi * freq * x + phase)
    return np.sign(raw)


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    Returns a sequence of signed step heights.
    Score is the upper bound on C3 (max |C3|), lower is better.
    """
    start_time = time.time()
    best_h = np.ones(N)
    best_score = _score(best_h)  # baseline: all ones

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    # Phase 1: Coarse grid search over frequency and phase for sine-based sign patterns
    # Frequencies: integer multiples give natural period structures
    freqs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 50]
    phases = np.linspace(0, np.pi, 50, endpoint=False)

    for freq in freqs:
        for phase in phases:
            if not elapsed_ok():
                return best_h
            h = _make_sine_pattern(freq, phase)
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()

    # Phase 2: Local refinement around best (freq, phase) with finer grid
    if elapsed_ok():
        # Determine best freq and phase from best_h by least-squares fit
        x = np.linspace(0, 1, N, endpoint=False)
        # Just do a finer local search around the current best parameters
        # Extract approximate freq/phase from best_h
        # For now, refine around a few nearby frequencies
        for dfreq in range(-2, 3):
            for base_freq in freqs:
                f = base_freq + dfreq * 0.5
                if f <= 0:
                    continue
                phases_fine = np.linspace(0, np.pi, 100, endpoint=False)
                for phase in phases_fine:
                    if not elapsed_ok():
                        return best_h
                    h = _make_sine_pattern(f, phase)
                    s = _score(h)
                    if s < best_score:
                        best_score = s
                        best_h = h.copy()
                if not elapsed_ok():
                    return best_h

    # Phase 3: Try block-structured patterns (e.g., +1+1-1-1, +1+1+1-1-1-1, etc.)
    if elapsed_ok():
        block_lengths = [2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100]
        for blen in block_lengths:
            for half in range(1, blen // 2 + 1):
                if not elapsed_ok():
                    return best_h
                # Pattern: first 'half' blocks +1, rest -1, repeating with period blen
                pattern = np.ones(blen)
                pattern[half:] = -1
                h = np.tile(pattern, N // blen + 1)[:N]
                s = _score(h)
                if s < best_score:
                    best_score = s
                    best_h = h.copy()

    # Phase 4: Greedy coordinate descent with random restarts
    if elapsed_ok():
        # Start from the best pattern found so far
        h_work = best_h.copy().astype(np.float64)
        h_work[np.where(h_work == 0)] = 1  # ensure no zeros
        
        # Greedy coordinate descent: try flipping each element, keep if it improves
        improved = True
        while improved and elapsed_ok():
            improved = False
            # Randomize the order of coordinates to break symmetry
            order = np.random.permutation(N)
            for idx in order:
                if not elapsed_ok():
                    break
                # Try flipping this element
                h_try = h_work.copy()
                h_try[idx] *= -1
                s_try = _score(h_try)
                if s_try < best_score - 1e-12:
                    best_score = s_try
                    best_h = h_try.copy()
                    h_work = h_try.copy()
                    improved = True
        
        # If time remains, do a few random restarts with local refinement
        for restart in range(5):
            if not elapsed_ok():
                break
            
            # Random initialization
            h_rand = np.random.choice([-1.0, 1.0], size=N)
            s_rand = _score(h_rand)
            
            if s_rand < best_score:
                best_score = s_rand
                best_h = h_rand.copy()
            
            # Local greedy refinement from random start
            h_local = h_rand.copy()
            local_improved = True
            while local_improved and elapsed_ok():
                local_improved = False
                order = np.random.permutation(N)
                for idx in order:
                    if not elapsed_ok():
                        break
                    h_try = h_local.copy()
                    h_try[idx] *= -1
                    s_try = _score(h_try)
                    if s_try < best_score - 1e-12:
                        best_score = s_try
                        best_h = h_try.copy()
                        h_local = h_try.copy()
                        local_improved = True

    # Final: ensure output is ±1 (signed step heights)
    best_h = np.sign(best_h).astype(np.float64)
    # If all zeros, default to ones
    if np.all(best_h == 0):
        best_h = np.ones(N)
    return best_h


# Quick self-test (not run by evaluator)
if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
