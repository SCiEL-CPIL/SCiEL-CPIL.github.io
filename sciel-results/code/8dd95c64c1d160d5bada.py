"""Fresh design: coordinate-descent / projected-subgradient minimax for Erdos overlap."""
import numpy as np
import time

N = 100
W = 2.0 / N  # width of each step

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 100
W = 2.0 / N


def _overlap_profile(h):
    """
    Return array O[k] = sum_i h[i] * (1 - h[(i - k) mod N]) * W
    for shift k in 0..N-1 (circular shift on the torus [0,2]).
    """
    one_minus_h = 1.0 - h
    # O[k] = sum_i h[i] * one_minus_h[(i - k) mod N]
    # = circular cross-correlation of h and one_minus_h
    # Using FFT for speed
    H = np.fft.rfft(h, n=2 * N)
    OmH = np.fft.rfft(one_minus_h, n=2 * N)
    # circular cross-correlation: ifft( H * conj(OmH) )
    corr = np.fft.irfft(H * np.conj(OmH), n=2 * N)
    # corr[j] = sum_i h[i] * one_minus_h[(i - j) mod (2N)]
    # We want shifts 0..N-1 within the N-periodic domain; take first N
    O = corr[:N] * W
    return O


def _max_overlap(h):
    O = _overlap_profile(h)
    return float(np.max(O))


def _project(h):
    """Project onto {h in [0,1]^N : sum h = N/2} approximately via clipping + scaling."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    s = h.sum()
    if s > 0:
        h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Initialize with a structured pattern: alternating high/low
    rng = np.random.default_rng(12345)
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)

    # Try a few structured initializations
    candidates = []
    # constant
    candidates.append(np.full(N, 0.5))
    # two-level alternating
    alt = np.empty(N)
    alt[0::2] = 0.0
    alt[1::2] = 1.0
    candidates.append(alt)
    # block pattern
    block = np.tile([0.0, 1.0, 0.0, 1.0], N // 4)
    candidates.append(block)
    # random
    candidates.append(rng.random(N))
    # sine wave
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(0.5 + 0.5 * np.sin(x))
    # ramp
    candidates.append(np.linspace(0, 1, N))

    for c in candidates:
        c = _project(c.copy())
        sc = _max_overlap(c)
        if sc < best_score:
            best_score = sc
            best_h = c.copy()

    # Coordinate descent with subgradient on the max-overlap objective
    h = best_h.copy()
    step = 0.05
    no_improve = 0
    max_iter = 20000

    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        O = _overlap_profile(h)
        k_max = int(np.argmax(O))
        # subgradient of O[k_max] w.r.t. h[i]:
        # O[k] = sum_i h[i] * (1 - h[(i-k) mod N]) * W
        # dO[k]/dh[i] = W * (1 - h[(i-k) mod N]) - W * h[(i+k) mod N]
        # (because h[i] appears in O[k] as h[i]*(1-h[i-k]) and as (1-h[i+k])*h[i])
        dO = np.empty(N)
        for i in range(N):
            a = (i - k_max) % N
            b = (i + k_max) % N
            dO[i] = W * (1.0 - h[a]) - W * h[b]
        # gradient of the constraint sum h = N/2 is uniform; we handle via projection
        h_new = h - step * dO
        h_new = _project(h_new)
        sc = _max_overlap(h_new)
        if sc < best_score - 1e-12:
            best_score = sc
            best_h = h_new.copy()
            no_improve = 0
            # adaptive step
            if it % 500 == 0 and it > 0:
                step = max(0.005, step * 0.95)
        else:
            no_improve += 1
            if no_improve > 300:
                step = max(0.005, step * 0.8)
                no_improve = 0
        h = h_new

    # Final projection to ensure exact sum
    best_h = _project(best_h)
    # Ensure sum is exactly N/2
    s = best_h.sum()
    if s != 0:
        best_h = best_h * (N / 2.0 / s)
    best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
# EVOLVE-BLOCK-END
