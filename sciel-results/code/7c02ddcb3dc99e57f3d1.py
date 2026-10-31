"""Signed step functions minimizing max|autoconv|/integral^2 via m-sequence construction."""

import numpy as np
import time

N = 400

_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)


def _m_sequence(k, seed_offset=0):
    """Generate an m-sequence of period 2^k - 1 using a primitive polynomial over GF(2).

    Returns a binary vector of length 2^k - 1.
    """
    # Primitive polynomials for small k (coefficients as int bitmask, x^k + ...)
    # Represented as: tap positions for the linear feedback.
    # For k=1..12, use known primitive polynomials.
    # We'll use a table of primitive polynomials in the form:
    # polynomial coefficients (excluding x^k term) as a list of tap indices.
    _primitive_polys = {
        1: [0],       # x + 1 (period 1, trivial)
        2: [0],       # x^2 + 1 (period 3)
        3: [0],       # x^3 + 1 (period 7)
        4: [1],       # x^4 + x + 1 (period 15)
        5: [0],       # x^5 + 1 (period 31)
        6: [1],       # x^6 + x + 1 (period 63)
        7: [0],       # x^7 + 1 (period 127)
        8: [4],       # x^8 + x^4 + x^3 + x^2 + 1 (period 255)
        9: [4],       # x^9 + x^4 + 1 (period 511)
        10: [3],      # x^10 + x^3 + 1 (period 1023)
        11: [0],      # x^11 + 1 (period 2047)
        12: [6],      # x^12 + x^6 + x^4 + x + 1 (period 4095)
    }

    if k not in _primitive_polys:
        # Fallback: use a random LFSR
        return _random_binary(2**k - 1)

    taps = _primitive_polys[k]
    period = 2**k - 1

    # LFSR state: k bits, represented as integer
    # Feedback: new_bit = XOR of bits at tap positions
    state = (1 << k) - 1  # all ones initial state (non-zero)
    # Apply seed_offset to vary the phase
    for _ in range(seed_offset % period):
        # Advance LFSR by one step
        feedback = 0
        for t in taps:
            feedback ^= (state >> t) & 1
        state = ((state << 1) | feedback) & ((1 << k) - 1)
        # Ensure state is non-zero (m-sequence never returns to zero with primitive poly)
        if state == 0:
            state = 1

    seq = np.zeros(period, dtype=np.int8)
    for i in range(period):
        seq[i] = state & 1
        feedback = 0
        for t in taps:
            feedback ^= (state >> t) & 1
        state = ((state << 1) | feedback) & ((1 << k) - 1)
        if state == 0:
            state = 1

    return seq


def _random_binary(n):
    """Generate a random binary sequence of length n."""
    return np.random.randint(0, 2, size=n, dtype=np.int8)


def _to_signed(seq):
    """Convert binary sequence to signed (+1/-1) sequence."""
    return 2 * seq.astype(np.float64) - 1


def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the autoconvolution of x."""
    n = len(x)
    # Autoconvolution via FFT
    f = np.fft.fft(x, n=2*n)
    ac = np.fft.ifft(f * f).real
    # The autoconvolution has length 2n-1, centered at n-1
    # ac[i] = sum_j x[j] * x[j+i-n+1] for appropriate ranges
    # Actually, np.fft.ifft(np.fft.fft(x)**2) gives circular autoconvolution
    # For linear autoconvolution, we need to handle the wrap-around
    # Simpler: use scipy or manual
    # Let's use the direct approach for correctness
    # ac[k] = sum_i x[i] * x[i+k] for k in -(n-1)..(n-1)
    # Using FFT: zero-pad to 2n-1
    N_fft = 2 * n - 1
    F = np.fft.fft(x, n=N_fft)
    ac_lin = np.fft.ifft(F * F).real
    # ac_lin[k] for k=0..N_fft-1 gives the linear autoconvolution
    # The peak is at k=0 (or k=n-1 depending on convention)
    # Actually, ifft(fft(x,2n-1)**2) gives circular conv of zero-padded x with itself
    # This equals the linear autoconvolution for the non-wrapped part
    return np.max(np.abs(ac_lin))


def _autoconv_max_abs_fast(x):
    """Faster computation using scipy or optimized FFT."""
    n = len(x)
    # Use scipy.signal.fftconvolve if available, else numpy
    try:
        from scipy.signal import fftconvolve
        ac = fftconvolve(x, x, mode='full')
        return np.max(np.abs(ac))
    except ImportError:
        N_fft = 1
        while N_fft < 2*n - 1:
            N_fft *= 2
        F = np.fft.fft(x, n=N_fft)
        ac = np.fft.ifft(F * F).real[:2*n-1]
        return np.max(np.abs(ac))


def _squared_integral(x):
    """Compute (sum x)^2."""
    s = np.sum(x)
    return s * s


def _score(x):
    """Compute max|autoconv| / (sum x)^2. Lower is better."""
    si = _squared_integral(x)
    if abs(si) < 1e-12:
        # If sum is zero, the ratio is undefined/infinite.
        # Return a large penalty.
        return 1e10
    mac = _autoconv_max_abs_fast(x)
    return mac / si


def _local_sign_flip_optimize(x, max_iter=5000):
    """Local sign-flip optimization to reduce max|autoconv|/integral^2."""
    x = x.copy().astype(np.float64)
    n = len(x)
    best_score = _score(x)
    current_score = best_score

    # Precompute autoconvolution
    def compute_ac(x_arr):
        N_fft = 1
        while N_fft < 2*n - 1:
            N_fft *= 2
        F = np.fft.fft(x_arr, n=N_fft)
        return np.fft.ifft(F * F).real[:2*n-1]

    ac = compute_ac(x)
    si = np.sum(x)
    si_sq = si * si

    # Track the peak lag
    peak_idx = np.argmax(np.abs(ac))
    peak_val = np.max(np.abs(ac))

    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break

        # Find the lag with max absolute autoconvolution
        peak_idx = np.argmax(np.abs(ac))
        peak_val = np.abs(ac[peak_idx])

        # If peak is at lag 0, that's the energy, hard to reduce without changing integral
        # Try to flip signs to reduce peak
        # For each position i, flipping x[i] changes ac by:
        # delta_ac[k] = -4 * x[i] * (sum over appropriate range)
        # This is complex. Instead, try random flips and accept improvements.

        # Try a batch of random sign flips
        improved = False
        for _ in range(10):
            idx = np.random.randint(n)
            x_new = x.copy()
            x_new[idx] = -x_new[idx]
            si_new = np.sum(x_new)
            si_sq_new = si_new * si_new
            if abs(si_sq_new) < 1e-12:
                continue
            ac_new = compute_ac(x_new)
            score_new = np.max(np.abs(ac_new)) / si_sq_new
            if score_new < current_score * 0.999:  # require meaningful improvement
                x = x_new
                ac = ac_new
                current_score = score_new
                improved = True
                break

        if not improved and it % 100 == 0:
            # Try multiple random flips
            for _ in range(5):
                idxs = np.random.choice(n, size=3, replace=False)
                x_new = x.copy()
                x_new[idxs] = -x_new[idxs]
                si_new = np.sum(x_new)
                si_sq_new = si_new * si_new
                if abs(si_sq_new) < 1e-12:
                    continue
                ac_new = compute_ac(x_new)
                score_new = np.max(np.abs(ac_new)) / si_sq_new
                if score_new < current_score * 0.999:
                    x = x_new
                    ac = ac_new
                    current_score = score_new
                    improved = True
                    break
            if not improved:
                break

    return x


def _build_m_sequence_candidate(k, seed_offset=0, pad_to=None):
    """Build a signed candidate from an m-sequence of order k."""
    period = 2**k - 1
    if pad_to is None:
        pad_to = N

    if period >= pad_to:
        # Take first N elements
        binary = _m_sequence(k, seed_offset)[:pad_to]
    else:
        # Tile and truncate
        num_tiles = (pad_to + period - 1) // period
        binary = np.tile(_m_sequence(k, seed_offset), num_tiles)[:pad_to]

    return _to_signed(binary)


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2.

    Uses m-sequences (maximum-length sequences) which have near-flat autocorrelation
    with sidelobes of magnitude -1 for all non-zero lags, structurally minimizing
    the peak-to-average ratio.
    """
    if _time_remaining() < 2.0:
        # Fallback: return ones
        return np.ones(N)

    best_x = None
    best_score = 1e15

    # Try different m-sequence orders
    # k=9 gives period 511 > 400, so we take first 400
    # k=8 gives period 255, tile to 400
    # k=10 gives period 1023, take first 400
    candidates = []

    # k=9: period 511, take first 400
    for offset in range(0, 10):
        if _time_remaining() < 5.0:
            break
        x = _build_m_sequence_candidate(9, seed_offset=offset)
        sc = _score(x)
        candidates.append((sc, x, f"mseq_k9_off{offset}"))

    # k=10: period 1023, take first 400
    for offset in range(0, 10):
        if _time_remaining() < 5.0:
            break
        x = _build_m_sequence_candidate(10, seed_offset=offset)
        sc = _score(x)
        candidates.append((sc, x, f"mseq_k10_off{offset}"))

    # k=8: period 255, tile to 400
    for offset in range(0, 10):
        if _time_remaining() < 5.0:
            break
        x = _build_m_sequence_candidate(8, seed_offset=offset)
        sc = _score(x)
        candidates.append((sc, x, f"mseq_k8_off{offset}"))

    # Also try Thue-Morse for comparison
    # Thue-Morse: t(n) = popcount(n) mod 2
    tm = np.array([(i & 1) ^ ((i >> 1) & 1) ^ ((i >> 2) & 1) ^ ((i >> 3) & 1) ^
                   ((i >> 4) & 1) ^ ((i >> 5) & 1) ^ ((i >> 6) & 1) ^ ((i >> 7) & 1)
                   for i in range(N)], dtype=np.int8)
    # Actually, proper Thue-Morse: t(n) = number of 1s in binary representation of n, mod 2
    def thue_morse(n):
        # Count bits mod 2
        c = 0
        while n:
            c ^= (n & 1)
            n >>= 1
        return c
    tm = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    x_tm = _to_signed(tm)
    sc_tm = _score(x_tm)
    candidates.append((sc_tm, x_tm, "thue_morse"))

    # Sort by score
    candidates.sort(key=lambda t: t[0])

    if not candidates:
        return np.ones(N)

    best_score, best_x, best_name = candidates[0]

    # Local optimization on the best candidate
    if _time_remaining() > 10.0:
        optimized = _local_sign_flip_optimize(best_x, max_iter=3000)
        opt_score = _score(optimized)
        if opt_score < best_score:
            best_x = optimized
            best_score = opt_score

    # If all candidates are bad (sum near zero), fall back
    if abs(np.sum(best_x)) < 1e-6:
        return np.ones(N)

    return best_x


# EVOLVE-BLOCK-END
