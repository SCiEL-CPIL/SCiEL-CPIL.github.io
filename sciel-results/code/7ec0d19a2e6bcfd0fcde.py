"""Rewritten program: off-diagonal flat-spectrum design via analytic-gradient descent.

Minimizes  max_k |c_k| / (sum x)^2  for the linear autoconvolution c = x * x.
The diagonal c_0 = sum(x^2) is the binding lower bound (>= 1 when sum x = 1),
so the real goal is to drive the OFF-diagonal autoconvolution toward zero
(flat magnitude spectrum / low crest factor). We optimize a soft-max surrogate
of the off-diagonal autoconvolution with an analytic FFT-based gradient.
"""
import time
import numpy as np

N = 400
_SIZE = 2 * N - 1
_FFT = 1
while _FFT < _SIZE:
    _FFT <<= 1


def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    F = np.fft.rfft(x, n=_FFT)
    conv = np.fft.irfft(F * F, n=_FFT)
    return conv[:_SIZE]


def _score_from_signal(x):
    """The raw metric: max(|autoconv|) / (sum x)^2. Lower is better."""
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    return np.max(np.abs(conv)) / (integral * integral)


def _offdiag_abs(conv):
    """|c_k| for k >= 1 (exclude the diagonal c_0)."""
    return np.abs(conv[1:])


def _soft_max_offdiag(x, temp):
    """Smooth surrogate for max_{k>=1} |c_k| (log-sum-exp)."""
    a = _offdiag_abs(_autoconv_full(x))
    m = np.max(a)
    if m < 1e-12:
        return 0.0
    return m + temp * np.log(np.sum(np.exp((a - m) / temp)))


def _grad_soft_offdiag(x, temp):
    """Analytic gradient of the soft-max off-diagonal autoconvolution."""
    F = np.fft.rfft(x, n=_FFT)
    conv = np.fft.irfft(F * F, n=_FFT)[:_SIZE]
    a = _offdiag_abs(conv)
    m = np.max(a)
    if m < 1e-12:
        return np.zeros(N)
    w = np.exp((a - m) / temp)
    w /= w.sum()
    # d a_k / d x = 2 * sign(c_k) * c_k, restricted to k >= 1
    grad_full = np.zeros(_SIZE)
    grad_full[1:] = 2.0 * np.sign(conv[1:]) * conv[1:]
    grad_full *= w
    # d/dx_k sum_j grad_full[j] x_{j-k} = 2 * sum_j grad_full[j] x_{j-k}
    grad = 2.0 * np.fft.irfft(np.fft.rfft(grad_full, n=_FFT) * F, n=_FFT)[:N]
    return grad


def _spectral_random_signal(rng, n_modes):
    """Random signed signal with a flat-ish magnitude spectrum."""
    kmax = N - 1
    K = np.arange(kmax + 1)
    mags = rng.uniform(0.5, 1.0, size=kmax + 1)
    ph = rng.uniform(-np.pi, np.pi, size=kmax + 1)
    ph[0] = 0.0
    F = np.zeros(kmax + 1, dtype=np.complex128)
    F[K <= n_modes] = mags[K <= n_modes] * np.exp(1j * ph[K <= n_modes])
    x = np.fft.irfft(F, n=N)
    s = np.sum(x)
    if abs(s) > 1e-12:
        x /= s
    return x


def _gradient_descent(x0, deadline, temp0=0.6, temp1=0.03, max_iter=4000):
    """Adaptive-temp analytic gradient descent on the off-diagonal soft-max."""
    x = x0.copy()
    s = np.sum(x)
    if abs(s) > 1e-12:
        x /= s
    best = _score_from_signal(x)
    best_x = x.copy()
    lr = 1e-3
    for it in range(max_iter):
        if time.time() > deadline:
            break
        temp = temp0 + (temp1 - temp0) * (it / max(1, max_iter))
        g = _grad_soft_offdiag(x, temp)
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-12:
            break
        x = x - lr * g
        s = np.sum(x)
        if abs(s) > 1e-12:
            x /= s
        sc = _score_from_signal(x)
        if sc < best:
            best = sc
            best_x = x.copy()
            lr = min(0.05, lr * 1.03)
        else:
            lr *= 0.8
    return best_x, best


def construct():
    deadline = time.time() + 135  # safety margin before the 150 s kill

    rng = np.random.default_rng(1234)
    best_x = np.ones(N)
    best_score = _score_from_signal(best_x)

    # A few diverse initial points: all-ones, random spectral, flat-spectrum design.
    inits = [np.ones(N)]
    for nm in (25, 40, 60, 80, 120, 160):
        inits.append(_spectral_random_signal(rng, nm))

    for x0 in inits:
        if time.time() > deadline - 5:
            break
        x, sc = _gradient_descent(x0, deadline)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Polish the global best with a longer, colder run.
    if time.time() < deadline - 5:
        x, sc = _gradient_descent(best_x, deadline, temp0=0.2, temp1=0.01, max_iter=6000)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    return best_x
