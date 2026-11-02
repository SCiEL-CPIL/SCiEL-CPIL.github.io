import numpy as np
import time

N = 400


def _autocorr(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    n = 2 * N
    F = np.fft.rfft(h, n=n)
    ac = np.fft.irfft(F * np.conj(F), n=n)[:N]
    return ac


def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score(h, beta):
    # smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 110.0          # safety margin inside the 120s limit
        rng = np.random.default_rng(20240517)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        cur_smooth = _smooth_score(h, 30.0)

        # ---------- simulated annealing with coordinated block sign-flips ----------
        T = 0.02
        T_min = 1e-4
        decay = 0.9995
        beta = 30.0
        sizes = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32])

        while time.time() < deadline - 3.0:
            if rng.random() < 0.75:
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                new_hard = _hard_score(h)
                new_smooth = _smooth_score(h, beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    h[sl] = old
            else:
                i = int(rng.integers(0, N))
                old = h[i]
                h[i] *= -1.0
                new_hard = _hard_score(h)
                new_smooth = _smooth_score(h, beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    h[i] = old
            T = max(T_min, T * decay)

        # ---------- greedy multi-scale local polish ----------
        h = best_h.copy()
        sizes_polish = [1, 2, 4, 8, 16, 32]
        for _pass in range(40):
            if time.time() > deadline - 1.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 1.0:
                        return best_h
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    new_hard = _hard_score(h)
                    if new_hard < best_hard - 1e-12:
                        best_hard = new_hard
                        best_h = h.copy()
                        any_improve = True
                    else:
                        h[sl] = old
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)
