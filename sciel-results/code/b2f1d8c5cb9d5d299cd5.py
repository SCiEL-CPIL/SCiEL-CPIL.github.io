"""Rewritten program: spectral + smooth-max QCQP for Erdos min-overlap."""
import time
import numpy as np

N = 100
_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _overlaps(h):
    """Circular autocorrelations O_k = sum_i h_i * h_{i-k} for k in 0..N-1."""
    # O_k = sum_i h_i h_{i-k}. Use FFT: circular cross-correlation of h with h.
    # corr[k] = sum_i h_i * h_{(i-k) mod N}.
    H = np.fft.fft(h)
    P = H * np.conj(H)          # power spectrum (real, non-negative)
    # corr[k] = ifft(P)[k]  gives sum_i h_i * h_{i-k}? Verify: ifft(|H|^2)[k] = sum_i h_i h_{(i-k)}
    corr = np.fft.ifft(P).real
    return corr


def _smooth_max(corr, lam):
    """Log-sum-exp smooth max of corr with scale 1/lam (lam large => closer to true max)."""
    m = corr.max()
    return m + np.log(np.sum(np.exp(lam * (corr - m)))) / lam


def _project(h):
    """Project onto {0<=h<=1, sum h = N/2}. Simple: clip then adjust mean via water-filling-ish."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        d = (target - s) / N
        # shift those not at bound
        h_new = h + d
        h_new = np.clip(h_new, 0.0, 1.0)
        if np.array_equal(h_new, h):
            # distribute d only among non-bound entries
            free = (h > 1e-9) & (h < 1 - 1e-9)
            if not free.any():
                break
            d2 = (target - h.sum()) / free.sum()
            h = h.copy()
            h[free] += d2
            h = np.clip(h, 0.0, 1.0)
        else:
            h = h_new
    # final exact mean fix on free set
    s = h.sum()
    free = (h > 1e-9) & (h < 1 - 1e-9)
    if free.any() and abs(h.sum() - target) > 1e-12:
        d = (target - h.sum()) / free.sum()
        h = h.copy()
        h[free] += d
        h = np.clip(h, 0.0, 1.0)
    return h


def _grad_smooth_max(h, lam):
    """Gradient of smooth max of corr w.r.t. h."""
    corr = _overlaps(h)
    m = corr.max()
    w = np.exp(lam * (corr - m))
    w = w / w.sum()  # softmax weights over k
    # d corr[k] / d h_j : corr[k] = sum_i h_i h_{i-k}
    # = sum_i h_i h_{i-k} + sum_i h_i h_{i-k} (symmetric) -> grad wrt h_j = 2 * sum_i [term where j involved]
    # corr[k] = sum_i h_i * h_{i-k}. d/dh_j: for i=j: h_{j-k}; for i-k=j => i=j+k: h_{j+k}.
    # So d corr[k]/d h_j = h_{j-k} + h_{j+k} (indices mod N).
    # grad_h_j = sum_k w_k * (h_{j-k} + h_{j+k})
    # This is 2 * (w * h) circular convolution at j? conv of w and h: (w*h)[j] = sum_k w_k h_{j-k}.
    # and sum_k w_k h_{j+k} = (w * h_rev)[j] where h_rev[i]=h[-i]. Since we want h_{j+k}=h_{j-(-k)}.
    W = np.fft.fft(w)
    Hh = np.fft.fft(h)
    conv1 = np.fft.ifft(W * Hh).real              # sum_k w_k h_{j-k}
    Hrev = np.fft.fft(h[::-1]) if False else np.fft.fft(np.roll(h, 1))  # placeholder
    # h_{j+k}: let r[i] = h_{-i mod N} = h[(-i) mod N]. r[i]=h[(-i)%N].
    r = np.array([h[(-i) % N] for i in range(N)])
    R = np.fft.fft(r)
    conv2 = np.fft.ifft(W * R).real               # sum_k w_k r_{j-k}=sum_k w_k h_{-(j-k)}=sum_k w_k h_{k-j}=sum_k w_k h_{j+k}? 
    # r_{j-k} = h_{-(j-k)} = h_{k-j} = h_{(k-j) mod N} = h_{j-k} reversed... let's just trust symmetry:
    # Actually d corr[k]/d h_j = h_{j-k} + h_{j+k}. Both are circular shifts.
    # sum_k w_k h_{j-k} = (w * h)[j]. sum_k w_k h_{j+k} = (w * h)[j] if w symmetric? w is symmetric (corr is real even? corr[k]=corr[-k] yes autocorr is even). So w symmetric => both equal.
    # corr is even: corr[k]=corr[N-k]. So w symmetric. Thus sum_k w_k h_{j+k} = sum_k w_k h_{j-k} (reindex).
    grad = 2.0 * conv1
    return grad


def _spectral_init():
    """Flat-magnitude FFT init: h = 0.5 + c * real part of a chirp, then project."""
    # Choose h whose |H| is as flat as possible. A chirp gives flat magnitude.
    idx = np.arange(N)
    phi = 2 * np.pi * (idx ** 2) / N
    base = np.cos(phi)
    base = base - base.mean()
    # scale so that after adding 0.5, within [0,1] and mean 0.5
    c = 0.45 / (np.abs(base).max() + 1e-12)
    h = 0.5 + c * base
    h = _project(h)
    return h


def _optimize(h0, lam_start=5.0, lam_end=200.0, iters=400, lr=0.02):
    h = h0.copy()
    best_h = h.copy()
    best_val = _overlaps(h).max()
    for it in range(iters):
        if _elapsed() > _DEADLINE - 5.0:
            break
        lam = lam_start * (lam_end / lam_start) ** (it / max(1, iters - 1))
        corr = _overlaps(h)
        val = corr.max()
        if val < best_val:
            best_val = val
            best_h = h.copy()
        g = _grad_smooth_max(h, lam)
        # projected gradient step
        h_new = h - lr * g
        h_new = _project(h_new)
        # line search (simple backtracking)
        f_cur = _smooth_max(_overlaps(h), lam)
        alpha = 1.0
        for _ in range(30):
            h_try = _project(h - alpha * lr * g)
            f_try = _smooth_max(_overlaps(h_try), lam)
            if f_try <= f_cur:
                h = h_try
                break
            alpha *= 0.5
        else:
            h = h_new
    # final polish with high lam
    h = best_h.copy()
    for it in range(200):
        if _elapsed() > _DEADLINE - 3.0:
            break
        lam = 500.0
        corr = _overlaps(h)
        if corr.max() < best_val:
            best_val = corr.max()
            best_h = h.copy()
        g = _grad_smooth_max(h, lam)
        h_new = _project(h - 0.005 * g)
        h = h_new
    return best_h, best_val


def construct():
    t0 = time.time()
    h0 = _spectral_init()
    h, val = _optimize(h0)
    # ensure exact sum
    h = _project(h)
    if _elapsed() > _DEADLINE - 2.0:
        return h
    return h
