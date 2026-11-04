"""Nonnegative step function on 600 equal intervals of [-1/4,1/4] minimizing
the normalized peak of its autoconvolution (an upper bound on C1)."""
import time
import numpy as np

N = 600
A = 1.0 / 4.0          # interval is [-A, A]
DUR = 150.0
MARGIN = 10.0          # safety margin before the 150 s kill


def _score(x):
    """Normalized peak of autoconvolution: max(autocorr)/sum(x)."""
    s = x.sum()
    if s <= 0:
        return 1.0
    ac = np.correlate(x, x, mode="full")
    return ac.max() / s


def _tent(half_width_frac=0.5, smooth=0.0):
    """Centered triangular bump. half_width_frac in (0,1]; smooth in [0,1]
    blends from triangle toward a cosine bump (smoother, slightly wider)."""
    n = np.arange(N)
    c = (n - (N - 1) / 2.0) / ((N - 1) / 2.0)      # in [-1, 1]
    w = half_width_frac
    tri = np.clip(1.0 - np.abs(c) / w, 0.0, None)
    cos = np.clip(0.5 * (1.0 + np.cos(np.pi * c / w)), 0.0, None)
    f = (1.0 - smooth) * tri + smooth * cos
    return np.maximum(f, 0.0)


def construct():
    t0 = time.time()
    best_x = None
    best_val = 1.0  # uniform baseline

    # Candidate family: vary width and smoothness.
    candidates = []
    for hw in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0):
        for sm in (0.0, 0.3, 0.6, 1.0):
            candidates.append((hw, sm))

    # Evaluate the whole family (cheap) and keep the best.
    for hw, sm in candidates:
        x = _tent(hw, sm)
        v = _score(x)
        if v < best_val:
            best_val = v
            best_x = x

    # Gradient-based refinement (projected gradient on the nonneg cone) to
    # further reduce the normalized peak autoconvolution.
    x = best_x.astype(float).copy()
    deadline = t0 + DUR - MARGIN

    def grad_of(x):
        # minimize max(autocorr)/sum(x); approximate with gradient of
        # (autocorr at peak index)/sum(x). Use the argmax shift.
        s = x.sum()
        ac = np.correlate(x, x, "full")
        k = int(np.argmax(ac))
        # d(ac_k)/dx_j = x_{k-j} (with appropriate indexing)
        # ac index k corresponds to lag (k - (N-1)).
        lag = k - (N - 1)
        # derivative w.r.t. x_j of ac_k = sum_j x_j x_{j-lag}
        # d/dx_i = x_{i-lag} + x_{i+lag}
        g = np.zeros(N)
        for i in range(N):
            a = i - lag
            b = i + lag
            if 0 <= a < N:
                g[i] += x[a]
            if 0 <= b < N:
                g[i] += x[b]
        # f = ac_k / s ; df = (g*s - ac_k*1)/s^2
        return (g * s - ac[k]) / (s * s)

    # Simple projected gradient descent with a few adaptive steps.
    step = 0.05
    it = 0
    while time.time() < deadline and it < 4000:
        it += 1
        g = grad_of(x)
        x = x - step * g
        x = np.maximum(x, 0.0)          # nonnegativity projection
        v = _score(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
        else:
            step *= 0.8                  # shrink step on no improvement
            if step < 1e-6:
                break

    if best_x is None:
        return [1.0] * N
    return list(np.asarray(best_x, dtype=float))


if __name__ == "__main__":
    r = construct()
    print(len(r), _score(np.array(r)))
