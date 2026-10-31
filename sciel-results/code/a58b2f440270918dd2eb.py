"""Signed step function minimizing max |autoconvolution sidelobe| / (sum x)^2."""
import numpy as np
import time

N = 400
DEADLINE = 132.0  # safety margin before the 150 s kill

_rng = np.random.default_rng(12345)


def _c_forward(x):
    """Forward linear autoconvolution lags 0..N-1 via FFT."""
    n = 2 * N - 1
    F = np.fft.fft(x, n=n)
    r = np.fft.ifft(F * np.conj(F)).real
    return r[:N]


def _metric(c, S):
    s2 = S * S
    if s2 <= 0:
        return float('inf')
    return float(np.max(np.abs(c[1:]))) / s2


def _flip(c, x, S, j):
    """Return (new_c, new_S) after flipping x[j], without mutating inputs."""
    v = x[j]
    term1 = np.concatenate([x[j + 1:], np.zeros(j)])          # x[j+k] for k=1..N-1
    term2 = np.concatenate([x[j - 1::-1], np.zeros(N - 1 - j)])  # x[j-k] for k=1..N-1
    delta = -2.0 * v * (term1 + term2)
    new_c = c.copy()
    new_c[1:] += delta
    new_S = S - 2 * v
    return new_c, new_S


def _sa(x0, budget):
    """Simulated annealing with incremental updates. Returns best (x, metric)."""
    x = x0.copy().astype(np.int8)
    c = _c_forward(x).astype(np.float64)
    S = int(x.sum())
    best_metric = _metric(c, S)
    best_x = x.copy()
    T = 5e-5
    T_end = 1e-8
    end = time.time() + budget
    it = 0
    while time.time() < end:
        j = _rng.integers(0, N)
        new_c, new_S = _flip(c, x, S, j)
        new_m = _metric(new_c, new_S)
        if new_m < best_metric:
            best_metric = new_m
            best_x = x.copy()
        d = new_m - _metric(c, S)
        if d < 0 or _rng.random() < np.exp(-d / max(T, T_end)):
            c, x, S = new_c, np.where(np.arange(N) == j, -x, x).astype(np.int8), new_S
        it += 1
        if it % 4096 == 0:
            T = max(T_end, T * 0.985)
            # periodic exact recompute to kill drift
            c = _c_forward(x).astype(np.float64)
            S = int(x.sum())
    return best_x, best_metric


def _polish(x, budget):
    """Steepest-descent: repeatedly apply the single best flip."""
    x = x.copy().astype(np.int8)
    c = _c_forward(x).astype(np.float64)
    S = int(x.sum())
    m = _metric(c, S)
    end = time.time() + budget
    while time.time() < end:
        best_j, best_m, best_c, best_S = None, m, None, None
        for j in range(N):
            nc, nS = _flip(c, x, S, j)
            nm = _metric(nc, nS)
            if nm < best_m - 1e-15:
                best_j, best_m, best_c, best_S = j, nm, nc, nS
        if best_j is None:
            break
        c, x, S = best_c, np.where(np.arange(N) == best_j, -x, x).astype(np.int8), best_S
        m = best_m
    return x, m


def construct():
    t0 = time.time()
    global_best_x = None
    global_best_m = float('inf')

    def _consider(x, m):
        nonlocal global_best_x, global_best_m
        if m < global_best_m:
            global_best_m = m
            global_best_x = x.copy()

    # Initialization 1: all ones (natural start; single flips already help)
    x1 = np.ones(N, dtype=np.int8)
    c1 = _c_forward(x1)
    _consider(x1, _metric(c1, int(x1.sum())))

    # Initialization 2 & 3: random +/-1
    x2 = _rng.choice([-1, 1], size=N).astype(np.int8)
    _consider(x2, _metric(_c_forward(x2), int(x2.sum())))
    x3 = _rng.choice([-1, 1], size=N).astype(np.int8)
    _consider(x3, _metric(_c_forward(x3), int(x3.sum())))

    # Annealing from each init, with a time slice each
    inits = [x1, x2, x3]
    for xi in inits:
        if time.time() > DEADLINE - 8:
            break
        budget = min(30.0, DEADLINE - (time.time() - t0) - 6.0)
        if budget <= 1.0:
            break
        bx, bm = _sa(xi, budget)
        _consider(bx, bm)

    # Polish the global best with whatever time remains
    if global_best_x is not None and time.time() < DEADLINE - 2.0:
        px, pm = _polish(global_best_x, max(0.5, DEADLINE - (time.time() - t0) - 1.0))
        _consider(px, pm)

    if global_best_x is None:
        return np.ones(N)
    # Final sanity: ensure finite, real metric
    S = int(global_best_x.sum())
    if S * S <= 0:
        return np.ones(N)
    return global_best_x
