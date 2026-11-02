import time
import numpy as np

N = 600
DEADLINE = 145.0

def construct():
    t0 = time.time()
    n = 2 * N
    # mask for sidelobes: exclude lag 0 and the wrap-around lag n-? 
    # linear autocorrelation lags 1..N-1 are at indices 1..N-1 in c
    mask = np.zeros(n, dtype=bool)
    mask[1:N] = True
    mask[n-(N-1):n] = True  # symmetric negative lags

    def autocorr(f):
        F = np.fft.fft(f, n=n)
        return np.fft.ifft(F * np.conj(F)).real

    def value(f):
        c = autocorr(f)
        return float(c[mask].max())

    best_f = np.ones(N)
    best_val = value(best_f)

    rng = np.random.default_rng(12345)
    beta = 20.0
    lr = 0.3

    def optimize(f):
        nonlocal beta, lr
        for it in range(300):
            if time.time() - t0 > DEADLINE:
                return
            c = autocorr(f)
            z = beta * c * c
            z = np.where(mask, z, -1e9)
            z = z - z[mask].max()
            w = np.exp(z)
            w = w / w.sum()
            dJ_dc = 2.0 * c * w
            F = np.fft.fft(f, n=n)
            grad = 4.0 * np.fft.ifft(np.fft.fft(dJ_dc, n=n) * F).real
            f = f - lr * grad
            f = np.maximum(f, 0.0)
            nrm = np.linalg.norm(f)
            if nrm < 1e-12:
                return
            f = f / nrm
        return f

    restarts = 0
    while time.time() - t0 < DEADLINE:
        f = rng.random(N)
        f = np.maximum(f, 0)
        nrm = np.linalg.norm(f)
        if nrm < 1e-12:
            continue
        f = f / nrm
        f = optimize(f)
        v = value(f)
        if v < best_val:
            best_val = v
            best_f = f.copy()
        restarts += 1
        beta = min(beta * 1.15, 2000.0)

    m = best_f.max()
    if m > 1e-12:
        best_f = best_f / m
    return [float(x) for x in best_f]
