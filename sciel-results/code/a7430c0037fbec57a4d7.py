"""Signed step function minimizing max |autoconv| / (sum a)^2 via sign-flip coordinate descent."""
import numpy as np
import time

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 5.0  # stop this many seconds before the kill limit


def _autoconv(a):
    """Full linear autoconvolution c[k] = sum_i a[i]*a[i+k], k in [0, N-1]."""
    # Use FFT for speed: pad to 2N-1
    n = 2 * N - 1
    fa = np.fft.rfft(a, n=n)
    c = np.fft.irfft(fa * fa.conj(), n=n)
    return c[:N]


def _max_abs_autoconv(a):
    c = _autoconv(a)
    return np.max(np.abs(c))


def _score_value(a):
    """Raw value = max |autoconv| / (sum a)^2. Lower is better."""
    s = np.sum(a)
    if s == 0.0:
        # Degenerate: integral zero -> ratio undefined; penalize heavily.
        return np.max(np.abs(_autoconv(a))) + 1e9
    return _max_abs_autoconv(a) / (s * s)


def _greedy_sign_flip(a, max_passes=200, deadline=None):
    """Coordinate descent: flip signs to reduce max |autoconv|/(sum)^2.
    We optimize the numerator (max |autoconv|) primarily; the denominator
    (sum)^2 also changes with flips, so we track the full ratio."""
    a = a.astype(np.float64).copy()
    c = _autoconv(a)
    cur_val = _score_value(a)
    best_a = a.copy()
    best_val = cur_val

    for p in range(max_passes):
        if deadline is not None and time.time() > deadline - SAFETY_MARGIN:
            break
        improved = False
        # Randomize order to avoid systematic bias
        order = np.random.permutation(N)
        for j in order:
            if deadline is not None and time.time() > deadline - SAFETY_MARGIN:
                break
            aj = a[j]
            if aj == 0.0:
                continue
            # Delta to autoconvolution from flipping a[j] -> -a[j]
            # c[k] gets -2*aj*(a[j+k] + a[j-k]) for valid indices (k!=0 handled by zeros)
            delta = np.zeros(N)
            # pairs (j, j+k): k in 0..N-1-j -> a[j]*a[j+k]
            if j + 1 < N:
                delta[:N - j] += -2.0 * aj * a[j + 1:]
            # pairs (j-k, j): k in 1..j -> a[j-k]*a[j]
            if j > 0:
                delta[:j] += -2.0 * aj * a[:j][::-1]
            # Note: k=0 term unchanged (a[j]^2 same), delta[0] stays 0.
            new_c = c + delta
            new_max = np.max(np.abs(new_c))
            new_s = np.sum(a) - 2.0 * aj
            if new_s == 0.0:
                new_val = new_max + 1e9
            else:
                new_val = new_max / (new_s * new_s)
            if new_val < cur_val - 1e-12:
                # Accept flip
                a[j] = -aj
                c = new_c
                cur_val = new_val
                improved = True
                if cur_val < best_val:
                    best_val = cur_val
                    best_a = a.copy()
        if not improved:
            break
    return best_a, best_val


def _make_seeds():
    seeds = []
    # 1. Random
    seeds.append(np.where(np.random.rand(N) < 0.5, 1.0, -1.0))
    # 2. Chirp: sign varies linearly in phase
    t = np.arange(N)
    phase = 2 * np.pi * (t / N) * 3.0
    seeds.append(np.sign(np.cos(phase)))
    # 3. Block: alternating blocks of length ~ N/8
    blen = max(1, N // 8)
    blk = []
    sgn = 1.0
    i = 0
    while i < N:
        blk.extend([sgn] * min(blen, N - i))
        sgn = -sgn
        i += blen
    seeds.append(np.array(blk[:N], dtype=np.float64))
    # 4. Sinusoidal sign
    t = np.arange(N)
    seeds.append(np.sign(np.sin(2 * np.pi * t / N * 2.0)))
    # 5. Random with bias toward +
    seeds.append(np.where(np.random.rand(N) < 0.6, 1.0, -1.0))
    # 6. High-frequency alternating
    seeds.append(np.where(np.arange(N) % 2 == 0, 1.0, -1.0))
    return seeds


def construct():
    deadline = time.time() + DEADLINE
    best_a = None
    best_val = np.inf

    # Evaluate baseline all-ones as a fallback
    ones = np.ones(N)
    base_val = _score_value(ones)
    best_a = ones.copy()
    best_val = base_val

    seeds = _make_seeds()
    for s in seeds:
        if time.time() > deadline - SAFETY_MARGIN:
            break
        a, val = _greedy_sign_flip(s, max_passes=150, deadline=deadline)
        if val < best_val:
            best_val = val
            best_a = a

    # If we still have time, do a few extra random restarts from the best
    extra = 0
    while time.time() < deadline - SAFETY_MARGIN - 3.0 and extra < 6:
        # Perturb best_a by flipping a random subset
        a = best_a.copy()
        mask = np.random.rand(N) < 0.3
        a[mask] *= -1.0
        a, val = _greedy_sign_flip(a, max_passes=100, deadline=deadline)
        if val < best_val:
            best_val = val
            best_a = a
        extra += 1

    return best_a
