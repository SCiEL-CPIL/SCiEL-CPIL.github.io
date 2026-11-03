N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_peak(h):
    h = np.asarray(h, dtype=float)
    n = len(h)
    size = 2 * n
    f = np.fft.fft(h, n=size)
    conv = np.fft.ifft(f * np.conj(f)).real
    return float(np.max(conv[:n]))

def _score(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    l1 = float(np.sum(h))
    if l1 <= 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = float(np.sum(h_norm * h_norm))
    peak = _autoconv_peak(h_norm)
    if peak <= 1e-15:
        return 0.0
    return l2_sq / peak

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_h = np.ones(N) / N
    best_score = _score(best_h)

    inits = [
        np.ones(N) / N,
        np.array([1.0] + [0.1] * (N - 1)),
        np.array([0.5] * (N // 2) + [0.0] * (N - N // 2)),
        np.random.dirichlet(np.ones(N)),
        np.random.dirichlet(np.ones(N) * 0.5),
        np.zeros(N),
    ]
    inits[5][N // 2] = 1.0
    rng = np.random.default_rng(42)
    inits.append(rng.dirichlet(np.ones(N) * 2.0))

    for init in inits:
        if time.time() > deadline - 5:
            break
        init = np.maximum(np.asarray(init, dtype=float), 0.0)
        s0 = _score(init)
        if s0 > best_score:
            best_h = init.copy()
            best_score = s0

    bounds = [(0.0, None)] * N
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

    for init in inits:
        if time.time() > deadline - 10:
            break
        init = np.maximum(np.asarray(init, dtype=float), 0.0)
        l1 = float(np.sum(init))
        if l1 > 1e-15:
            init = init / l1
        else:
            init = np.ones(N) / N
        try:
            res = minimize(lambda x: -_score(x), init, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 150, 'ftol': 1e-9})
            h_res = np.maximum(res.x, 0.0)
            s_res = _score(h_res)
            if s_res > best_score:
                best_h = h_res.copy()
                best_score = s_res
        except Exception:
            pass

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            old = best_h[i]
            candidates = {0.0, old, old * 0.5, old * 0.7, old * 1.3, old * 2.0, 1.0 / N}
            if best_h.sum() > 0:
                candidates.update(np.quantile(best_h, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist())
            best_val = old
            best_s = best_score
            for c in candidates:
                c = max(0.0, c)
                h_test = best_h.copy()
                h_test[i] = c
                s_test = _score(h_test)
                if s_test > best_s:
                    best_s = s_test
                    best_val = c
            if best_val != old:
                best_h[i] = best_val
                best_score = best_s
                improved = True

    if time.time() < deadline - 2:
        for i in range(N):
            old = best_h[i]
            for factor in (0.8, 0.9, 1.1, 1.2):
                h_test = best_h.copy()
                h_test[i] = max(0.0, old * factor)
                s_test = _score(h_test)
                if s_test > best_score:
                    best_h = h_test
                    best_score = s_test

    return list(np.maximum(best_h, 0.0))
# EVOLVE-BLOCK-END
