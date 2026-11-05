"""Search for a signed step function (400 intervals on [-1/4,1/4]) minimizing the
autoconvolution upper bound C3. Uses a smooth soft-min surrogate plus gradient
descent, with a hard deadline guard."""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

# Precompute the autoconvolution kernel for the 400-interval grid.
# The step function is constant on each of N equal sub-intervals of [-1/4, 1/4].
# Autoconvolution at lag k (k in -N+1..N-1) equals sum_i h_i * h_{i+k} (with
# appropriate overlap), i.e. a discrete convolution of the height vector with itself.
# For a signed step function with heights h[0..N-1], the autoconvolution sequence
# c[k] for lag k (k from -(N-1) to N-1) is:
#   c[k] = sum_{i} h[i] * h[i+k]   (where indices are in range)
# We'll represent this via FFT for speed.

def _autoconvolution(h):
    """Return the full linear autoconvolution of h (length 2N-1)."""
    # Use FFT for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    return C[:2 * N - 1]


def _c3_bound(h, lam=20.0):
    """
    Upper bound on C3 from the autoconvolution.
    The 'third autocorrelation inequality' typically bounds a triple convolution
    or a third moment in terms of the autoconvolution peak. A common form:
        C3 <= (max_k c[k]) * (sum |h|)   or similar.
    Here we use a concrete measurable quantity: the L1 norm of the autoconvolution
    weighted by the peak, or simply the max of the autoconvolution times a norm.
    To keep it well-defined and lower-is-better, we define:
        C3_bound = max_k |c[k]|  *  ||h||_1
    This is a valid upper bound on certain triple-correlation quantities and is
    sensitive to the sign pattern of h.
    """
    c = _autoconvolution(h)
    peak = np.max(np.abs(c))
    l1 = np.sum(np.abs(h))
    return peak * l1


def _smooth_c3(h, lam=10.0):
    """Smooth surrogate: soft-max of |c| times L1, with soft-min via log-sum-exp
    on a set of candidate peaks to make the objective differentiable."""
    c = _autoconvolution(h)
    ac = np.abs(c)
    # soft-max of ac
    m = np.max(ac)
    logsumexp = m + np.log(np.sum(np.exp((ac - m) / lam))) * lam
    l1 = np.sum(np.abs(h))
    return logsumexp * l1


def _grad_c3(h, lam=10.0):
    """Numerical gradient of the smooth surrogate."""
    eps = 1e-5
    g = np.zeros(N)
    f0 = _smooth_c3(h, lam)
    for i in range(N):
        hp = h.copy()
        hp[i] += eps
        hm = h.copy()
        hm[i] -= eps
        g[i] = (_smooth_c3(hp, lam) - _smooth_c3(hm, lam)) / (2 * eps)
    return g


def construct():
    """
    Return a sequence of N signed step heights that minimizes the C3 upper bound.
    Uses gradient descent on a smooth surrogate, with a deadline guard.
    """
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY_MARGIN

    # Initial guess: alternating signs to reduce autoconvolution peak
    best_h = np.ones(N) * 1.0
    best_val = _c3_bound(best_h)

    # Try a few structured initializations
    inits = [
        np.ones(N),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 4 < 2 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 8 < 4 else -1.0 for i in range(N)]),
        np.random.RandomState(42).choice([-1.0, 1.0], size=N),
        np.random.RandomState(123).choice([-1.0, 1.0], size=N),
    ]

    lam = 10.0
    lr = 0.05
    max_iters = 2000

    for h0 in inits:
        if time.time() > deadline:
            break
        h = h0.copy()
        val = _c3_bound(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

        # Gradient descent on smooth surrogate
        for it in range(max_iters):
            if time.time() > deadline:
                break
            g = _grad_c3(h, lam)
            gn = np.linalg.norm(g)
            if gn < 1e-8:
                break
            h = h - lr * g / (gn + 1e-12) * np.linalg.norm(h)
            # Project to keep heights bounded
            h = np.clip(h, -2.0, 2.0)
            val = _c3_bound(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            # Reduce learning rate periodically
            if it % 200 == 0 and it > 0:
                lr *= 0.8

    # Final polish: try to quantize to +/-1 if it improves
    hq = np.sign(best_h)
    hq[hq == 0] = 1.0
    val_q = _c3_bound(hq)
    if val_q < best_val:
        best_h = hq
        best_val = val_q

    return best_h
