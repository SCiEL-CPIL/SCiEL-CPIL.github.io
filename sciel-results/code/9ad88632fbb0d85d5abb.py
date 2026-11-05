"""Rewritten program: multi-start Thue-Morse + FFT-guided local search
to minimize the autoconvolution peak (upper bound on C3)."""
import time
import numpy as np

N = 400

# --- deadline guard -------------------------------------------------------
_T_START = time.time()
_DEADLINE = _T_START + 140.0  # 10s safety margin under the 150s kill


def _time_left():
    return _DEADLINE - time.time()


# --- autocorrelation via FFT ----------------------------------------------
def _autocorr_max(x):
    """Return (max_abs_nonzero_lag, full autocorrelation array).
    x is a real array of length N.
    """
    n = len(x)
    # zero-pad to next power of two >= 2n for linear autocorrelation
    m = 1
    while m < 2 * n:
        m <<= 1
    F = np.fft.rfft(x, m)
    acf = np.fft.irfft(F * np.conj(F), m)[:n]
    # acf[0] = sum x^2 = n (for ±1). Max over nonzero lags:
    # lags 1..n-1 are acf[1:]; negative lags are acf[1:] mirrored
    peak = float(np.max(np.abs(acf[1:]))) if n > 1 else 0.0
    return peak, acf


def _bound_value(x):
    """The raw 'upper bound on C3' we minimize. Lower is better.
    Defined as max |autocorrelation| over nonzero lags, normalized so the
    all-ones sequence (which has peak = N-1) corresponds to the seed value 1.0
    after the evaluator's seed/value normalization. We return the peak itself;
    the evaluator handles normalization. For internal comparison we use peak.
    """
    peak, _ = _autocorr_max(x)
    return peak


# --- Thue-Morse generators -------------------------------------------------
def _thue_morse(n, start_bit=0, flip=False, shift=0):
    """Generate a ±1 Thue-Morse-like sequence of length n.
    Uses the parity of popcount of (i + shift) as the base sign.
    """
    idx = np.arange(n) + shift
    # popcount parity via XOR-fold trick (vectorized)
    v = idx.astype(np.uint64)
    v ^= v >> 32
    v ^= v >> 16
    v ^= v >> 8
    v ^= v >> 4
    v ^= v >> 2
    v ^= v >> 1
    parity = (v & 1).astype(np.int64)
    # map parity 0->+1, 1->-1
    seq = 1 - 2 * parity
    if start_bit == 1:
        seq = -seq
    if flip:
        seq = -seq
    return seq


def _random_signed(rng, n):
    return rng.choice([-1, 1], size=n)


# --- local search ----------------------------------------------------------
def _local_search(x, rng, max_iter, time_budget):
    """Hill-climb by single-bit flips minimizing the autocorrelation peak.
    Uses a steepest-descent-ish scan over a random subset of positions each
    iteration for speed, with occasional full scans.
    """
    n = len(x)
    x = x.copy()
    cur = _bound_value(x)
    best_x = x.copy()
    best_val = cur
    no_improve = 0
    i = 0
    while i < max_iter and time_budget() > 0:
        # choose a subset of positions to try this iteration
        if i % 10 == 0:
            # full scan every 10th iteration
            candidates = np.arange(n)
        else:
            k = min(n, 64)
            candidates = rng.choice(n, size=k, replace=False)
        improved = False
        for p in candidates:
            # try flipping p
            old = x[p]
            x[p] = -old
            val = _bound_value(x)
            if val < cur - 1e-12:
                cur = val
                improved = True
                if cur < best_val:
                    best_val = cur
                    best_x = x.copy()
            else:
                x[p] = old
        i += 1
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 30:
                # perturb to escape local minima
                k = min(n, 8)
                ps = rng.choice(n, size=k, replace=False)
                x[ps] = -x[ps]
                cur = _bound_value(x)
                no_improve = 0
    return best_x, best_val


# --- main construct --------------------------------------------------------
def construct():
    rng = np.random.default_rng(12345)
    starts = []

    # Thue-Morse variants (different shifts, flips, start bits)
    for shift in range(0, 8):
        for flip in (False, True):
            for sb in (0, 1):
                starts.append(_thue_morse(N, start_bit=sb, flip=flip, shift=shift))
    # A few structured random starts
    for _ in range(6):
        starts.append(_random_signed(rng, N))

    best_x = None
    best_val = np.inf

    for s in starts:
        if _time_left() < 3.0:
            break
        # quick local search per start with a per-start budget
        per_start_budget = lambda: _time_left()
        xs, vs = _local_search(s, rng, max_iter=400, time_budget=per_start_budget)
        if vs < best_val:
            best_val = vs
            best_x = xs
        if _time_left() < 3.0:
            break

    # If nothing was found (shouldn't happen), fall back to Thue-Morse
    if best_x is None:
        best_x = _thue_morse(N)

    # Ensure it's a plain numpy int array of ±1
    return np.asarray(best_x, dtype=np.int64)
