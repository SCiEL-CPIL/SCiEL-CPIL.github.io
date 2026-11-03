import numpy as np
import time

N = 600
_START = time.time()
_DEADLINE = 142.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(f):
    n = f.shape[0]
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    F = np.fft.rfft(f, n=m)
    ac = np.fft.irfft(F * np.conj(F), n=m)[:2 * n - 1]
    peak = ac[0]
    if peak <= 1e-300:
        return 1.0
    return float(np.max(ac[1:n]) / peak)

def _smooth_init(kind="gauss"):
    n = N
    c = (n - 1) / 2.0
    x = (np.arange(n) - c) / c
    if kind == "gauss":
        f = np.exp(-1.5 * x**2)
    elif kind == "supergauss":
        f = np.exp(-2.5 * x**2)
    elif kind == "hann":
        f = np.clip(0.5 * (1 + np.cos(np.pi * x)), 0, None)
    else:
        f = np.ones(n)
    f = np.clip(f, 0, None)
    if f.sum() <= 0:
        f = np.ones(n)
    f = f / f.max()
    return f

def _local_search(f, max_sweeps=None):
    f = np.array(f, dtype=float)
    best = _ratio(f)
    n = f.shape[0]
    step = 0.12
    sweeps = 0
    while _time_left() > 4.0:
        sweeps += 1
        if max_sweeps is not None and sweeps >= max_sweeps:
            break
        order = np.random.permutation(n)
        improved = False
        for i in order:
            if _time_left() < 4.0:
                break
            # try +/- step
            for sgn in (1.0, -1.0):
                val = f[i] + sgn * step
                if val < 0:
                    val = 0.0
                if val == f[i]:
                    continue
                f[i] = val
                r = _ratio(f)
                if r < best - 1e-12:
                    best = r
                    improved = True
                    break
                else:
                    f[i] = f[i] - sgn * step  # revert
        if improved:
            step = max(0.008, step * 0.9)
        else:
            step = max(0.008, step * 0.65)
            if step <= 0.009:
                step = 0.04  # kick
    return f, best

def _restarts(f0, n_restarts):
    best_f, best_r = f0.copy(), _ratio(f0)
    for _ in range(n_restarts):
        if _time_left() < 8.0:
            break
        f = f0.copy()
        mask = np.random.rand(N) < 0.25
        f[mask] = np.clip(f[mask] * (1 + 0.6 * np.random.randn(int(mask.sum()))), 0, None)
        if f.sum() <= 0:
            continue
        f, r = _local_search(f, max_sweeps=150)
        if r < best_r:
            best_f, best_r = f, r
    return best_f, best_r

def construct():
    np.random.seed(0)
    best_f, best_r = None, np.inf

    for kind in ("gauss", "supergauss", "hann"):
        if _time_left() < 8.0:
            break
        f0 = _smooth_init(kind)
        f, r = _local_search(f0, max_sweeps=300)
        if r < best_r:
            best_f, best_r = f, r

    if _time_left() > 12.0:
        f0 = _smooth_init("gauss")
        f, r = _restarts(f0, n_restarts=5)
        if r < best_r:
            best_f, best_r = f, r

    if _time_left() > 4.0 and best_f is not None:
        best_f, best_r = _local_search(best_f, max_sweeps=500)

    out = np.asarray(best_f, dtype=float)
    out = np.clip(out, 0, None)
    if not np.all(np.isfinite(out)):
        out = np.ones(N)
    return out.tolist()
