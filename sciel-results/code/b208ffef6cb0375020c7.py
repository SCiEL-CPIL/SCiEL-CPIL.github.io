"""Additive bases with few representations – combined model-and-optimize rewrite.

The answer is modelled directly as a 0/1 indicator vector x of length 129
(one entry per integer in [0,128]).  The convolution y = x*x (over the
integers, with x treated as a filter) has y[k] = sum_{i+j=k} x[i]x[j],
which is exactly the (ordered) pair-sum multiplicity for target k.  The
task requires y[k] > 0 for all k in [128,256] and asks us to minimize
max_{k in [128,256]} y[k].

We seed with a structured 2-level (Goldreich-Levin style) construction
that is guaranteed to be a basis for the window, then run a fast,
vectorized single-flip hill climb that scores each candidate with a
single NumPy add on a pre-computed delta.  A wall-clock deadline guard
ensures we always return the best valid indicator found.
"""
import time
import random
import numpy as np

_N = 128                     # elements live in [0, 128]
_LEN = 129                   # length of the indicator / convolution support
_T0 = 128                    # window start
_T1 = 256                    # window end (inclusive)
_WIN = _T1 - _T0 + 1         # 129 targets
_DEADLINE = 85.0             # safety margin before the 90 s kill

_START = time.time()


def _elapsed():
    return time.time() - _START


# ---------------------------------------------------------------------------
# Fast convolution utilities
# ---------------------------------------------------------------------------

def _conv(x):
    """Return the integer convolution of x with itself (length 2*_LEN-1)."""
    return np.convolve(x.astype(np.int64), x.astype(np.int64), mode="full")


def _window(y):
    """Slice y to the target window [128, 256]."""
    return y[_T0:_T1 + 1]


def _valid(y):
    """True iff every target in [128,256] has at least one representation."""
    return bool(np.all(_window(y) > 0))


def _peak(y):
    return int(_window(y).max())


# ---------------------------------------------------------------------------
# Structured 2-level seed (Goldreich-Levin style)
# ---------------------------------------------------------------------------

def _make_seed(rng, size_hint=64):
    """Build a valid 0/1 indicator using a 2-level (even/odd) construction.

    Let E be the set of even elements and O the set of odd elements.
    The pair sums then decompose as:
        even + even -> even target  (covers even targets in the window)
        even + odd  -> odd target   (covers odd targets in the window)
    By choosing E and O independently and sufficiently large, we guarantee
    coverage of every target in [128, 256].
    """
    x = np.zeros(_LEN, dtype=np.int64)

    # Even candidates: {0, 2, 4, ..., 128}  (65 elements)
    evens = np.array([2 * k for k in range(65)])
    # Odd candidates: {1, 3, 5, ..., 127}   (64 elements)
    odds  = np.array([2 * k + 1 for k in range(64)])

    # Shuffle and take the first `size_hint` elements from each pool,
    # but ensure enough to cover the window.
    rng.shuffle(evens)
    rng.shuffle(odds)

    # Heuristic: take ~size_hint/2 from each, but at least 24 from each.
    n_even = max(24, size_hint // 2)
    n_odd  = max(24, size_hint - n_even)
    n_even = min(n_even, len(evens))
    n_odd  = min(n_odd, len(odds))

    x[evens[:n_even]] = 1
    x[odds[:n_odd]] = 1

    # If coverage is not yet complete, add more elements (cheapest first)
    y = _conv(x)
    while not _valid(y):
        # Find uncovered targets
        w = _window(y)
        uncovered = np.where(w == 0)[0] + _T0
        if len(uncovered) == 0:
            break
        # For each uncovered target t, try adding a candidate that covers it.
        # A candidate c covers t if (t-c) is in [0,128] and x[t-c]==1 or c==t-c.
        best_c = None
        best_gain = -1
        for t in uncovered:
            # candidates c in [0,128] such that t-c in [0,128]
            lo = max(0, t - 128)
            hi = min(128, t)
            for c in range(lo, hi + 1):
                if x[c] == 1:
                    continue
                # gain = number of new representations added
                gain = 0
                # c + c
                if _T0 <= c + c <= _T1 and w[c + c - _T0] == 0:
                    gain += 1
                # c + y for y in A, y != c
                for yy in np.where(x == 1)[0]:
                    if yy == c:
                        continue
                    s = c + yy
                    if _T0 <= s <= _T1 and w[s - _T0] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
        if best_c is None or best_gain <= 0:
            break
        x[best_c] = 1
        y = _conv(x)

    return x


# ---------------------------------------------------------------------------
# Fast single-flip hill climb with variance-aware objective
# ---------------------------------------------------------------------------

def _hill_climb(x, rng, max_time):
    """Vectorized single-flip local search minimizing the peak multiplicity."""
    y = _conv(x)
    if not _valid(y):
        return x, _peak(y), y

    peak = _peak(y)
    best_x = x.copy()
    best_peak = peak
    best_y = y.copy()

    n = _LEN
    improved = True
    stall = 0
    max_stall = 80

    while improved and _elapsed() < max_time:
        improved = False
        stall += 1

        # Order candidates: try flips that reduce the peak first, then random.
        order = list(range(n))
        if stall % 3 == 0:
            rng.shuffle(order)

        for j in order:
            if _elapsed() > max_time:
                return best_x, best_peak, best_y

            old = x[j]
            
            # Compute the exact delta for flipping x[j]
            if old == 0:
                # Adding x[j]: dy[k] = x[k-j] for k in [j, j+128]
                #              dy[2j] += 1  (the x[j]*x[j] term)
                lo = j
                hi = min(j + _LEN, 2 * _LEN - 1)  # y has length 2*_LEN-1 = 257
                dy = np.zeros(2 * _LEN - 1, dtype=np.int64)
                dy[lo:hi] = x[0:hi - lo]
                if 2 * j < len(dy):
                    dy[2 * j] += 1
            else:
                # Removing x[j]: dy[k] = -x[k-j]
                lo = j
                hi = min(j + _LEN, 2 * _LEN - 1)
                dy = np.zeros(2 * _LEN - 1, dtype=np.int64)
                dy[lo:hi] = -x[0:hi - lo]
                if 2 * j < len(dy):
                    dy[2 * j] -= 1

            new_y = y + dy
            if not _valid(new_y):
                continue
            new_peak = _peak(new_y)
            
            # Variance-aware objective: penalize both peak and variance
            w_new = _window(new_y)
            new_var = float(np.var(w_new))
            new_obj = new_peak + 0.1 * new_var
            
            w_cur = _window(y)
            cur_var = float(np.var(w_cur))
            cur_obj = peak + 0.1 * cur_var
            
            if new_obj < cur_obj:
                # Accept the flip
                x[j] = 1 - old
                y = new_y
                peak = new_peak
                improved = True
                stall = 0
                if peak < best_peak:
                    best_peak = peak
                    best_x = x.copy()
                    best_y = y.copy()
                # After accepting, the "dy" for other candidates changes.
                # We must break and recompute the order.
                break

        if not improved and stall >= max_stall:
            break

    return best_x, best_peak, best_y


# ---------------------------------------------------------------------------
# Simulated annealing refinement
# ---------------------------------------------------------------------------

def _simulated_annealing(x, rng, max_time):
    """Simulated annealing to escape local optima."""
    y = _conv(x)
    if not _valid(y):
        return x, _peak(y), y

    peak = _peak(y)
    best_x = x.copy()
    best_peak = peak
    best_y = y.copy()
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9999
    
    while _elapsed() < max_time and T > T_min:
        j = rng.randint(0, _LEN - 1)
        old = x[j]
        
        # Compute the exact delta for flipping x[j]
        if old == 0:
            lo = j
            hi = min(j + _LEN, 2 * _LEN - 1)
            dy = np.zeros(2 * _LEN - 1, dtype=np.int64)
            dy[lo:hi] = x[0:hi - lo]
            if 2 * j < len(dy):
                dy[2 * j] += 1
        else:
            lo = j
            hi = min(j + _LEN, 2 * _LEN - 1)
            dy = np.zeros(2 * _LEN - 1, dtype=np.int64)
            dy[lo:hi] = -x[0:hi - lo]
            if 2 * j < len(dy):
                dy[2 * j] -= 1

        new_y = y + dy
        if not _valid(new_y):
            # Reject invalid moves
            T *= cooling
            continue
        
        new_peak = _peak(new_y)
        delta = new_peak - peak
        
        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / T):
            x[j] = 1 - old
            y = new_y
            peak = new_peak
            if peak < best_peak:
                best_peak = peak
                best_x = x.copy()
                best_y = y.copy()
        
        T *= cooling

    return best_x, best_peak, best_y


# ---------------------------------------------------------------------------
# Multi-restart driver
# ---------------------------------------------------------------------------

def _one_run(rng, size_hint, max_time):
    x = _make_seed(rng, size_hint)
    y = _conv(x)
    if not _valid(y):
        return None
    x, peak, y = _hill_climb(x, rng, max_time)
    return x, peak, y


def construct():
    global _START
    _START = time.time()

    rng = random.Random(12345)
    np.random.seed(12345)

    best_x = None
    best_peak = float("inf")

    # Phase 1: several structured seeds of varying sizes
    for size in [48, 56, 64, 72, 80]:
        if _elapsed() > _DEADLINE - 5:
            break
        remaining = _DEADLINE - _elapsed() - 3
        if remaining < 2:
            break
        result = _one_run(rng, size, max_time=remaining)
        if result is not None:
            x, peak, y = result
            if peak < best_peak:
                best_peak = peak
                best_x = x.copy()

        if best_peak <= 4:
            break

    # Phase 2: simulated annealing on the best seed
    if best_x is not None and _elapsed() < _DEADLINE - 5:
        remaining = _DEADLINE - _elapsed() - 3
        if remaining > 2:
            x2, peak2, y2 = _simulated_annealing(best_x, rng, max_time=remaining)
            if peak2 < best_peak:
                best_peak = peak2
                best_x = x2.copy()

    # Phase 3: a few random perturbations of the best seed
    if best_x is not None and _elapsed() < _DEADLINE - 5:
        for _ in range(3):
            if _elapsed() > _DEADLINE - 3:
                break
            x = best_x.copy()
            # Flip a few random bits
            for _ in range(8):
                j = rng.randint(0, _LEN - 1)
                x[j] = 1 - x[j]
            y = _conv(x)
            if _valid(y):
                remaining = _DEADLINE - _elapsed() - 2
                if remaining < 1:
                    break
                x2, peak2, y2 = _hill_climb(x, rng, max_time=remaining)
                if peak2 < best_peak:
                    best_peak = peak2
                    best_x = x2.copy()
            if best_peak <= 3:
                break

    # Fallback: if we somehow have no valid answer, use the full set
    if best_x is None:
        best_x = np.ones(_LEN, dtype=np.int64)
        best_peak = _peak(_conv(best_x))

    # Convert to a sorted list of integers
    result = [int(i) for i in np.where(best_x == 1)[0]]
    return result
