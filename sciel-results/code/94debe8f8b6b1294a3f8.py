"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _overlaps(h):
    """Return array of circular overlaps: ov[k] = sum_i h_i * (1 - h_{(i+k) mod N})."""
    N = len(h)
    one_minus_h = 1.0 - h
    ov = np.empty(N)
    # Vectorize with FFT: circular cross-correlation of h and one_minus_h
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    # circular correlation: corr[k] = sum_i h_{(i+k) mod N} * one_minus_h_i
    # We want sum_i h_i * one_minus_h_{(i+k) mod N} = sum_i h_{(i+k) mod N} * one_minus_h_{(i+k) mod N - k}? 
    # Simpler: corr[k] = sum_i a_{(i+k)} b_i = ifft( A * conj(B) ) where A=fft(a), B=fft(b)
    # Let a = one_minus_h, b = h. Then corr[k] = sum_i one_minus_h_{(i+k)} * h_i = ov[k].
    corr = np.fft.ifft(H * np.conj(G)).real
    ov = corr
    # Guard against tiny numerical errors
    return ov


def _max_overlap(h):
    """Maximum circular overlap of h with shifted (1-h)."""
    return float(np.max(_overlaps(h)))


def _project(x):
    """Project x onto {h : 0<=h<=1, sum(h)=N/2} via bisection on Lagrange multiplier."""
    N = len(x)
    target = N / 2.0

    def sum_at(lam):
        return np.sum(np.clip(x - lam, 0.0, 1.0))

    # Find lam range
    lo, hi = -10.0, 10.0
    # Expand if needed
    while sum_at(lo) < target:
        lo -= 1.0
    while sum_at(hi) > target:
        hi += 1.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = sum_at(mid)
        if s > target:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    h = np.clip(x - lam, 0.0, 1.0)
    # Final exact correction on free (non-clipped) coordinates
    free = (h > 0.0) & (h < 1.0)
    if np.any(free):
        diff = target - np.sum(h)
        h[free] += diff / np.sum(free)
        h = np.clip(h, 0.0, 1.0)
    return h


def _soft_max(ov, temperature):
    """Soft approximation of max(ov)."""
    m = np.max(ov)
    return m + temperature * np.log(np.sum(np.exp((ov - m) / temperature)))


def _objective(h, temperature):
    """Soft max overlap (smooth surrogate of the max)."""
    ov = _overlaps(h)
    return _soft_max(ov, temperature)


def _gradient(h, temperature):
    """
    Gradient of soft_max(ov) w.r.t. h.
    ov[k] = sum_i h_i * (1 - h_{(i+k) mod N})
    d ov[k] / d h_j = (1 - h_{(j+k) mod N}) - h_{(j-k) mod N}
    soft_max gradient: ov_weight[k] = exp((ov[k]-m)/T)/sum(...)
    grad_j = sum_k ov_weight[k] * ((1 - h_{(j+k)}) - h_{(j-k)})
    """
    N = len(h)
    ov = _overlaps(h)
    m = np.max(ov)
    w = np.exp((ov - m) / temperature)
    w = w / np.sum(w)

    # grad_j = sum_k w[k] * (1 - h_{(j+k)}) - sum_k w[k] * h_{(j-k)}
    # term1_j = sum_k w[k] * h_{(j+k)}  (circular convolution of w with h)
    # term2_j = sum_k w[k] * h_{(j-k)}  (circular correlation)
    # Use FFT for both
    W = np.fft.fft(w)
    H = np.fft.fft(h)
    # term1_j = sum_k w[k] h_{(j+k)} : circular cross-correlation
    term1 = np.fft.ifft(np.conj(W) * H).real
    # term2_j = sum_k w[k] h_{(j-k)} : circular convolution
    term2 = np.fft.ifft(W * H).real
    grad = np.sum(w) - term1 - term2
    # note sum(w)=1, so grad = 1 - term1 - term2
    return grad


def _fix_sum(h):
    """Ensure sum is exactly N/2 via projection (keeps box feasibility)."""
    return _project(h)


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    np.random.seed(12345)

    best_h = None
    best_score = np.inf

    def consider(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
            return
        s = _max_overlap(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # --- Candidate initializations ---
    inits = []
    # constant
    inits.append(np.full(N, 0.5))
    # sinusoids
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7]:
        inits.append(0.5 + 0.5 * np.sin(freq * t))
        inits.append(0.5 + 0.5 * np.cos(freq * t))
    # alternating
    alt = np.zeros(N); alt[0::2] = 1.0
    inits.append(alt)
    # random
    for _ in range(8):
        r = np.random.rand(N)
        inits.append(r)

    for h0 in inits:
        h0 = np.clip(h0, 0.0, 1.0)
        h0 = _fix_sum(h0)
        consider(h0)

    # --- Gradient descent with soft max, per-start refinement ---
    # Refine the best few starts
    starts = []
    for h0 in inits:
        h0 = np.clip(h0, 0.0, 1.0)
        h0 = _fix_sum(h0)
        starts.append(h0)
    # Add the current best
    if best_h is not None:
        starts.append(best_h.copy())

    temperature = 0.05
    eta = 0.02

    for h_start in starts:
        if time.time() > deadline:
            break
        h = h_start.copy()
        cur_score = _max_overlap(h)

        for it in range(3000):
            if time.time() > deadline:
                break
            grad = _gradient(h, temperature)
            h = h - eta * grad
            h = _fix_sum(h)
            new_score = _max_overlap(h)
            if new_score < cur_score:
                cur_score = new_score
                consider(h)
                # anneal temperature slowly for sharper max
                temperature = max(0.01, temperature * 0.999)
            else:
                # revert and shrink step
                h = h_start.copy() if (it % 50 == 0) else h + eta * grad
                eta *= 0.9
                temperature = max(0.01, temperature * 0.98)
                if eta < 1e-5:
                    break

    # --- Final local polish: coordinate-wise line search on the true max ---
    if best_h is not None:
        h = best_h.copy()
        cur = _max_overlap(h)
        polish_deadline = deadline - 5.0
        for _ in range(2000):
            if time.time() > polish_deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > polish_deadline:
                    break
                # try small moves in +/- directions
                for delta in [0.05, 0.02, 0.01, -0.05, -0.02, -0.01]:
                    cand = h.copy()
                    cand[i] = np.clip(cand[i] + delta, 0.0, 1.0)
                    cand = _fix_sum(cand)
                    s = _max_overlap(cand)
                    if s < cur - 1e-12:
                        h = cand
                        cur = s
                        improved = True
                        break
            if not improved:
                break
        consider(h)

    if best_h is None:
        # Fallback: constant 0.5 (always valid)
        best_h = np.full(N, 0.5)

    # Final exact sum enforcement
    best_h = _fix_sum(best_h)
    return best_h
# EVOLVE-BLOCK-END
