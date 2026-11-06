"""Third autocorrelation inequality - minimize the autocorrelation peak."""
import time
import numpy as np

N = 400
DEADLINE = 135.0          # safety margin under the 150 s kill


def _conv_max(h):
    """Exact objective: abs(2*N*max(full autoconv)/sum(h)**2)."""
    s = float(np.sum(h))
    if s == 0.0:
        return float('inf')
    c = np.convolve(h, h, mode='full')
    return abs(2.0 * N * float(np.max(c)) / (s * s))


def _conv_max_center(h):
    """Smooth objective: 2*N*max(conv) with sum(h) fixed to 1 (minimize toward 2)."""
    c = np.convolve(h, h, mode='full')
    return 2.0 * N * float(np.max(c))


def _soft_max(c, T):
    """log-sum-exp soft-max (smooth upper bound on max)."""
    m = np.max(c)
    return m + T * np.log(np.sum(np.exp((c - m) / T)))


def _grad_soft(c, T):
    m = np.max(c)
    e = np.exp((c - m) / T)
    e = e / np.sum(e)
    return e


def _dconv(h):
    """d(conv)/d(h_i) = 2*(h * rev(h))[i], computed once via FFT (O(N))."""
    f = np.fft.rfft(h, 2 * N - 1)
    g = np.fft.irfft(f * np.conj(f), 2 * N - 1)
    return 2.0 * g


def _raised_cosine():
    x = np.linspace(0.0, 1.0, N)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * x))


def _line_search(h, d, D, cur):
    """Backtracking line search on the exact peak value (sum kept = 1)."""
    best = cur
    best_h = h
    alpha = 0.5
    for _ in range(12):
        h2 = h + alpha * d
        s = float(np.sum(h2))
        if s == 0.0:
            alpha *= 0.5
            continue
        h2 = h2 / s
        v = _conv_max_center(h2)
        if v < best:
            best = v
            best_h = h2
        alpha *= 0.5
    return best_h, best


def _coord_descent(h, iters, budget):
    h = h / float(np.sum(h))
    for it in range(iters):
        if time.time() > budget:
            break
        c = np.convolve(h, h, mode='full')
        D = _dconv(h)
        cur = float(np.max(c))
        idx = int(np.argmax(c))
        step = D[idx]
        if step == 0.0:
            continue
        d = -np.sign(step) * np.ones(N)
        d[idx] -= (N - 1) * np.sign(step)   # keep sum(d)=0 -> sum unchanged
        h, cur = _line_search(h, d, D, cur)
    return h


def _soft_polish(h, budget):
    h = h / float(np.sum(h))
    T = 1e-3
    for it in range(400):
        if time.time() > budget:
            break
        c = np.convolve(h, h, mode='full')
        D = _dconv(h)
        w = _grad_soft(c, T)
        d = -w
        d -= np.mean(d)                      # keep sum=1
        h2 = h + 0.05 * d
        s = float(np.sum(h2))
        if s == 0.0:
            break
        h2 = h2 / s
        c2 = np.convolve(h2, h2, mode='full')
        if _soft_max(c2, T) < _soft_max(c, T):
            h = h2
    return h


def construct():
    t0 = time.time()
    budget = t0 + DEADLINE
    h = _raised_cosine().astype(float)
    h = h / float(np.sum(h))

    # Phase 1: coordinate descent on the exact (nonsmooth) peak.
    h = _coord_descent(h, iters=6000, budget=budget)

    # Phase 2: smooth (log-sum-exp) surrogate polish from a differentiable view.
    if time.time() < budget - 5.0:
        h = _soft_polish(h, budget)

    # Phase 3: high-precision exact polish (tighter line search, more iters).
    if time.time() < budget - 3.0:
        h = _coord_descent(h, iters=4000, budget=budget)

    h = h / float(np.sum(h))
    return h


if __name__ == '__main__':
    hh = construct()
    print("value =", _conv_max(hh), " score =", 2.0 / _conv_max(hh))
