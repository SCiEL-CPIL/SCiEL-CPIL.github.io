"""Frequency-domain design for the first autocorrelation inequality task.

We represent the nonnegative step function (600 equal intervals on [-1/4,1/4])
by a low-order cosine series (symmetric about 0). This enforces global smoothness
and lets us optimize a small number of continuous coefficients with a gradient
method, directly minimizing the normalized peak of the autoconvolution (C1).
"""
import time
import numpy as np

N = 600
# interval width
w = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # 0.5/600

# grid centers of the N intervals on [-1/4, 1/4]
_x = -0.25 + (np.arange(N) + 0.5) * w
_x = _x.astype(np.float64)


def _fourier_matrix(nharm):
    # cosine basis k=0..nharm-1 ; symmetric => only even harmonics matter, but we
    # use all and let the optimizer push odd ones to ~0 (harmless).
    k = np.arange(nharm)
    M = np.cos(2.0 * np.pi * k[None, :] * _x[:, None])  # (N, nharm)
    return M


def _autoconv_peak(f):
    # discrete autoconvolution via FFT (zero-padded to avoid wrap-around)
    n = N
    # convolution length 2n-1
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, n=size)
    conv = np.fft.irfft(F * F, n=size)[:2 * n - 1]
    return conv.max()


def _score(f):
    # normalized maximum of autoconvolution
    peak = _autoconv_peak(f)
    energy = f.sum()  # proportional to integral
    if energy <= 0:
        return np.inf
    return peak / energy


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s

    nharm = 12
    M = _fourier_matrix(nharm)

    # start from a broad symmetric bump: a=1, others 0
    a = np.zeros(nharm)
    a[0] = 1.0

    def f_of(a):
        return M @ a

    def obj_and_grad(a):
        f = f_of(a)
        # clip for gradient but keep differentiable-ish: use raw f for peak
        peak = _autoconv_peak(f)
        energy = f.sum()
        if energy <= 0:
            return np.inf, np.zeros(nharm)
        val = peak / energy
        # subgradient: peak is attained at index i*
        conv_n = 2 * N - 1
        size = 1
        while size < conv_n:
            size <<= 1
        F = np.fft.rfft(f, n=size)
        conv = np.fft.irfft(F * F, n=size)[:conv_n]
        i = int(np.argmax(conv))
        # d conv[i] / d a_j = sum_k f[k]*f[i-k] contribution -> use conv gradient
        # d conv / d f = 2 * (convolution of f with f) reflected; for fixed i:
        # d conv[i]/d f[m] = f[i-m] + f[m-i] (over valid range)
        # build g_f (length N)
        g_f = np.zeros(N)
        for m in range(N):
            a1 = i - m
            a2 = m - i
            s = 0.0
            if 0 <= a1 < N:
                s += f[a1]
            if 0 <= a2 < N:
                s += f[a2]
            g_f[m] = s
        dval_df = (g_f / energy) - (val / energy) * np.ones(N)
        grad = M.T @ dval_df
        return val, grad

    # simple projected gradient with backtracking, plus random restarts
    best_a = a.copy()
    best_val = _score(f_of(best_a))

    def project(a):
        # ensure resulting f nonnegative: we project in f-space then map back
        f = f_of(a)
        f = np.clip(f, 0.0, None)
        # least squares back to a (M may be ill-conditioned; use lstsq)
        a2, *_ = np.linalg.lstsq(M, f, rcond=None)
        return a2

    rng = np.random.default_rng(12345)
    n_restarts = 6
    max_iter = 400
    lr0 = 0.05

    for r in range(n_restarts):
        if r > 0:
            a = best_a.copy()
            # perturb
            a = a + rng.normal(0, 0.3, size=nharm)
            a = project(a)
        else:
            a = np.zeros(nharm)
            a[0] = 1.0
            # small random bump
            a += rng.normal(0, 0.1, size=nharm)
            a = project(a)

        lr = lr0
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            val, grad = obj_and_grad(a)
            if not np.isfinite(val):
                a = project(a)
                continue
            if val < best_val:
                best_val = val
                best_a = a.copy()
            # step
            step = a - lr * grad
            step = project(step)
            # backtracking on objective
            new_val = _score(f_of(step))
            if np.isfinite(new_val) and new_val < val:
                a = step
                lr = min(lr * 1.05, 0.5)
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break

    f = f_of(best_a)
    f = np.clip(f, 0.0, None)
    # normalize? score uses peak/sum so scaling cancels; keep as-is but ensure nonneg
    # Return sequence of N nonnegative heights
    return [float(v) for v in f]
