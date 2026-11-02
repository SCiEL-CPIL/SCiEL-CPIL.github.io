"""Erdos minimum overlap: choose step heights h in [0,1], sum=N/2, minimizing
max overlap of h with shifted (1-h). Uses smoothed subgradient descent with
water-filling projection and a deadline-guarded polish."""
import time
import numpy as np

N = 100
DEADLINE = 150.0
SAFETY = 15.0          # stop this many seconds before the hard kill
RNG_SEED = 12345


def _overlap_matrix():
    """A[k, i] = overlap of h with (1-h) shifted by k, as a linear form in h.
    overlap_k = sum_i h_i (1-h_{i-k})  (periodic wrap).
    = sum_i h_i - sum_i h_i h_{i-k}
    = N/2 - <h, h_{shifted by k}>.
    The constant N/2 is the same for all k, so maximizing overlap_k is equivalent
    to minimizing <h, h_{shifted by k}>. We build the quadratic form matrices."""
    # We'll represent overlap_k = N/2 - h @ Q_k @ h where Q_k is the circulant
    # with Q_k[i, j] = 1 if j == i - k (mod N).
    Qs = []
    for k in range(N):
        Q = np.zeros((N, N), dtype=float)
        for i in range(N):
            Q[i, (i - k) % N] = 1.0
        Qs.append(Q)
    return Qs


# Precompute once
_QS = _overlap_matrix()
_QS = np.stack(_QS)  # shape (N, N, N)


def _overlaps(h):
    """Return vector of overlap_k for all shifts k (length N)."""
    # overlap_k = N/2 - sum_i h_i h_{i-k}
    # vectorize: for each k, s_k = h @ Q_k @ h = sum_i h_i * h_{(i-k)%N}
    # h_{(i-k)%N} = np.roll(h, k)
    hs = np.stack([np.roll(h, k) for k in range(N)])  # (N, N)
    s = np.sum(h[None, :] * hs, axis=1)               # (N,)
    return N / 2.0 - s


def _soft_max(vals, beta):
    """Smooth max via log-sum-exp (scaled)."""
    m = np.max(vals)
    z = np.sum(np.exp(beta * (vals - m)))
    return m + np.log(z) / beta


def _soft_max_grad(vals, beta):
    m = np.max(vals)
    e = np.exp(beta * (vals - m))
    return e / np.sum(e)


def _project(h, total):
    """Project h onto {x in [0,1]^N : sum x = total} via bisection water-filling."""
    lo, hi = -1.0, 1.0
    def f(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))
    # f is decreasing in tau
    if f(lo) < total:
        return np.clip(h - lo, 0.0, 1.0)
    if f(hi) > total:
        return np.clip(h - hi, 0.0, 1.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > total:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    return np.clip(h - tau, 0.0, 1.0)


def _grad_overlaps(h):
    """Gradient of overlap_k w.r.t. h. overlap_k = N/2 - sum_i h_i h_{i-k}.
    d/dh_j = -h_{j-k} - h_{j+k} (by symmetry of the circulant, since
    sum_i h_i h_{i-k} = sum_j h_j (h_{j-k} + h_{j+k})/... let's be careful).
    s_k = sum_i h_i h_{(i-k)%N}. d s_k / d h_j: term i=j gives h_{(j-k)%N};
    term i such that (i-k)%N = j => i = (j+k)%N gives h_j. So
    d s_k/d h_j = h_{(j-k)%N} + h_{(j+k)%N}.
    Thus d overlap_k/d h_j = -(h_{(j-k)%N} + h_{(j+k)%N})."""
    G = np.zeros((N, N), dtype=float)
    for k in range(N):
        G[k, :] = -(np.roll(h, k) + np.roll(h, -k))
    return G  # (N, N): rows are k


def construct():
    t0 = time.time()
    total = N / 2.0
    rng = np.random.default_rng(RNG_SEED)

    # Start from a symmetric perturbation around 0.5
    best_h = np.full(N, 0.5)
    best_h = _project(best_h, total)
    best_ov = _overlaps(best_h)
    best_val = np.max(best_ov)

    # Smoothed subgradient descent on F = soft_max(beta * overlaps)
    beta = 20.0
    lr0 = 0.05
    for restart in range(3):
        if time.time() > DEADLINE - SAFETY:
            break
        # random-ish symmetric start
        if restart == 0:
            h = best_h.copy()
        else:
            base = np.full(N, 0.5)
            # low-frequency perturbation to break symmetry
            freq = rng.integers(1, N // 2 + 1)
            phase = rng.uniform(0, 2 * np.pi)
            amp = 0.3
            h = base + amp * np.cos(2 * np.pi * freq * np.arange(N) / N + phase)
            h = _project(h, total)

        F = _soft_max(_overlaps(h), beta)
        prev = F
        lr = lr0
        for it in range(4000):
            if time.time() > DEADLINE - SAFETY:
                break
            ov = _overlaps(h)
            w = _soft_max_grad(ov, beta)          # (N,)
            G = _grad_overlaps(h)                 # (N, N)
            grad = w @ G                          # (N,)
            step = lr * grad
            h_new = h - step
            h_new = _project(h_new, total)
            F_new = _soft_max(_overlaps(h_new), beta)
            if F_new < F:
                h = h_new
                F = F_new
                lr = min(0.1, lr * 1.01)
            else:
                lr *= 0.8
                if lr < 1e-6:
                    break
        # accept if improved
        cur_ov = _overlaps(h)
        cur_val = np.max(cur_ov)
        if cur_val < best_val - 1e-9:
            best_val = cur_val
            best_h = h.copy()

    # High-precision polish: minimize the exact (nonsmooth) max with projected
    # subgradient, very small steps, many iterations, on the best point.
    if time.time() < DEADLINE - SAFETY:
        h = best_h.copy()
        cur = np.max(_overlaps(h))
        lr = 1e-4
        for it in range(20000):
            if time.time() > DEADLINE - SAFETY:
                break
            ov = _overlaps(h)
            idx = np.argmax(ov)
            G = _grad_overlaps(h)
            grad = G[idx]
            h_new = _project(h - lr * grad, total)
            new = np.max(_overlaps(h_new))
            if new < cur - 1e-12:
                h = h_new
                cur = new
                lr = min(1e-3, lr * 1.02)
            else:
                lr *= 0.7
                if lr < 1e-9:
                    break
        best_h = h
        best_val = cur

    # Final exact projection to guarantee sum and bounds
    best_h = _project(best_h, total)
    # Force exact sum by tiny correction on a free (interior) entry
    s = best_h.sum()
    if abs(s - total) > 1e-12:
        diff = total - s
        # adjust an interior entry
        interior = np.where((best_h > 1e-6) & (best_h < 1 - 1e-6))[0]
        if len(interior) > 0:
            j = interior[0]
            best_h[j] += diff
            best_h = np.clip(best_h, 0.0, 1.0)
        else:
            # distribute
            best_h += diff / N
            best_h = np.clip(best_h, 0.0, 1.0)
    return best_h
