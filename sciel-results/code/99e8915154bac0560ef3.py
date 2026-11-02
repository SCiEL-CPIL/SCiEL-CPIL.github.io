"""Search for a low-autocorrelation signed step function (400 equal intervals).

Goal: minimise the peak aperiodic autocorrelation sidelobe of a signed (+-1)
sequence of length 400; this drives the C3 autoconvolution upper bound below
the all-ones baseline.  The seed program uses simulated annealing with block
flips and a greedy single-flip polish.

Refinements in this version (same design family, no switch):
  1.  Variable-length block-flip moves during annealing (1..24) instead of
      a fixed 1..11 window, giving better exploration of the 2^400 space.
  2.  After each anneal run, a short greedy *pair*-flip polish (flip two
      adjacent bits simultaneously) is interleaved with single-flip polish,
      helping escape flat plateaus that single flips cannot cross.
  3.  A final high-precision single-flip polish pass with a tighter time
      slice and an extra "second sweep" if time remains, tightening the
      local minimum.
  4.  Deadline guard: all loops check elapsed time against a hard 135 s
      wall-clock limit with a 5 s safety margin; the best result found so
      far is returned even if time expires mid-iteration.
  5.  Incremental autocorrelation update for single-flip evaluations: O(N)
      instead of O(N log N) per flip, enabling many more greedy sweeps.
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
# Annealing with variable-length block flips
# ---------------------------------------------------------------------------

def _anneal(seed, time_budget):
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-7
    start = time.time()
    while time.time() - start < time_budget:
        # Variable block length: longer blocks early, shorter later
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(24.0 * (1.0 - 0.5 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        new = _peak_sidelobe(h)
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99997)
    return best_h, best


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

        # Phase 1: long initial anneal from a random +-1 start.
        budget = min(30.0, _remaining() - 8.0)
        if budget > 2.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Short pair+single polish right after
            sub = min(5.0, _remaining() - 8.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: multiple shorter restarts for diversity.
        restart_id = 1
        while _remaining() > 12.0:
            sub = min(12.0, _remaining() - 12.0)
            if sub < 1.5:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Quick polish after each restart
            sub2 = min(3.0, _remaining() - 12.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: final high-precision single-flip polish.
        remaining = _remaining() - 2.0
        if remaining > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: one more polish pass if time remains.
        remaining2 = _remaining() - 1.0
        if remaining2 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
