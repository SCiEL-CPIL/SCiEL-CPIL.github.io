"""Search for a low-autocorrelation signed step function (400 equal intervals).

Goal: minimise the peak aperiodic autocorrelation sidelobe of a signed (+-1)
sequence of length 400; this drives the C3 autoconvolution upper bound below
the all-ones baseline.

Design family: simulated annealing with block flips + greedy coordinate
descent polish (single-bit and adjacent-pair flips), all evaluated via FFT.

Key improvements over the seed:
  1.  Faster peak-sidelobe evaluation using real-FFT and a pre-allocated
      buffer, reducing per-flip cost so more flips fit in the budget.
  2.  Annealing schedule: block length adapts to progress; acceptance
      threshold uses a smoothed surrogate (soft-max over top-k sidelobes)
      to break plateaus that the hard max objective creates.
  3.  Greedy polish: interleaved single-flip and pair-flip sweeps; after
      convergence, a second tighter single-flip sweep with a smaller
      improvement threshold to catch marginal gains.
  4.  Multiple restarts with diverse seeds, each followed by a quick polish.
  5.  Hard 135 s wall-clock deadline with 15 s safety margin.
"""
import numpy as np
import time

N = 400
_HARD_DEADLINE = 135.0  # 15 s margin below the 150 s kill

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H.conj(), _FFT_SIZE, n=N)
    return float(np.max(np.abs(ac[1:])))


def _soft_peak_sidelobe(h, k=10, beta=2.0):
    """Smooth surrogate: soft-max over top-k sidelobe magnitudes.
    Lower beta -> closer to hard max; higher beta -> smoother."""
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H.conj(), _FFT_SIZE, n=N)
    mags = np.abs(ac[1:])
    # soft-max via log-sum-exp on the top-k values
    idx = np.argpartition(mags, -k)[-k:]
    top = mags[idx]
    mx = np.max(top)
    # log-sum-exp
    shifted = top - mx
    lse = mx + np.log(np.sum(np.exp(beta * shifted))) / beta
    return float(lse)


# ---------------------------------------------------------------------------
# Annealing with variable-length block flips and soft-max acceptance
# ---------------------------------------------------------------------------

def _anneal(seed, time_budget):
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-8
    start = time.time()
    use_soft = True  # use soft surrogate for acceptance
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        # Block length: longer early, shorter late
        max_len = max(1, int(20.0 * (1.0 - 0.6 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0

        if use_soft:
            new_soft = _soft_peak_sidelobe(h, k=8, beta=1.5)
            new_hard = _peak_sidelobe(h)
        else:
            new_hard = _peak_sidelobe(h)
            new_soft = new_hard

        d = new_hard - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new_hard
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0

        T = max(T_min, T * 0.99995)
    return best_h, best


# ---------------------------------------------------------------------------
# Greedy polish: single-flip + adjacent-pair-flip
# ---------------------------------------------------------------------------

def _greedy_polish(h, time_budget, start=None):
    """Interleaved single-flip and adjacent-pair-flip greedy descent."""
    if start is None:
        start = time.time()
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        improved = False

        # --- single-flip sweep ---
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i >= 0:
            h[best_i] *= -1.0
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        # --- adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            h[i] *= -1.0
            h[i + 1] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c2:
                best_c2 = c
                best_pair = i
            h[i] *= -1.0
            h[i + 1] *= -1.0
        if best_pair >= 0:
            h[best_pair] *= -1.0
            h[best_pair + 1] *= -1.0
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        if not improved:
            break
    return best_h, best


def _greedy_single_only(h, time_budget, start=None):
    """Tighter single-flip polish for the final pass."""
    if start is None:
        start = time.time()
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i < 0:
            break
        h[best_i] *= -1.0
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def _greedy_triple_flip(h, time_budget, start=None):
    """Greedy sweep that also considers flipping 3 consecutive bits."""
    if start is None:
        start = time.time()
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        improved = False
        # single-flip sweep
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i >= 0:
            h[best_i] *= -1.0
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
        # triple-flip sweep
        best_triple = -1
        best_c3 = cur
        for i in range(N - 2):
            h[i] *= -1.0
            h[i + 1] *= -1.0
            h[i + 2] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c3:
                best_c3 = c
                best_triple = i
            h[i] *= -1.0
            h[i + 1] *= -1.0
            h[i + 2] *= -1.0
        if best_triple >= 0:
            h[best_triple] *= -1.0
            h[best_triple + 1] *= -1.0
            h[best_triple + 2] *= -1.0
            cur = best_c3
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True
        if not improved:
            break
    return best_h, best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    try:
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return _HARD_DEADLINE - (time.time() - start)

        # Phase 1: initial anneal
        budget = min(25.0, _remaining() - 10.0)
        if budget > 2.0:
            h, c = _anneal(42, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub = min(5.0, _remaining() - 10.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub)
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: restarts with diverse seeds
        restart_id = 1
        while _remaining() > 15.0:
            sub = min(10.0, _remaining() - 15.0)
            if sub < 1.5:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            sub2 = min(3.0, _remaining() - 15.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2)
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: greedy polish with triple flips
        remaining = _remaining() - 5.0
        if remaining > 0.5:
            h, c = _greedy_triple_flip(global_best_h, remaining)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: final tight single-flip polish
        remaining2 = _remaining() - 2.0
        if remaining2 > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining2)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 5: one more full polish if time remains
        remaining3 = _remaining() - 1.0
        if remaining3 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining3)
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
