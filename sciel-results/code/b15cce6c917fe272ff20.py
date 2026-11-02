"""Fresh design: low-autocorrelation signed step functions via sign-flip search."""
import time
import numpy as np

N = 400
_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _autocorr(x):
    """Full linear autocorrelation (lag -N+1..N-1) via FFT. x is real."""
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)
    # ac[0] = sum x^2 (lag 0); lags 1..n-1 are ac[1..]; negative lags are ac[2n-1..]
    return ac


def _score(x):
    """Raw value to minimize: max |autoconv| over all non-zero lags, divided by (sum x)^2.
    If sum x is ~0, use max|ac_nonzero| / N as a robust fallback (still lower is better)."""
    ac = _autocorr(x)
    n = len(x)
    # non-zero lags: indices 1..n-1 and 2n-(n-1)..2n-1  =>  1..n-1 and n+1..2n-1
    nz = np.concatenate([ac[1:n], ac[n + 1:2 * n]])
    peak = float(np.max(np.abs(nz)))
    s = float(np.sum(x))
    denom = s * s
    if denom < 1e-9:
        # fallback normalization; keep ratio comparable
        return peak / float(N)
    return peak / denom


def _greedy_flip(x, max_iters, deadline, start_score):
    """Iterative sign-flip: repeatedly flip the index that most reduces the score."""
    n = len(x)
    x = x.astype(np.float64).copy()
    best = start_score
    best_x = x.copy()
    # Precompute autocorr; updating full autocorr per flip is O(n log n). We sample candidate
    # flips to keep it fast.
    for it in range(max_iters):
        if _elapsed() > deadline:
            break
        ac = _autocorr(x)
        # For a sign flip at index i, the change in autocorr is a rank-1 update:
        # ac_new(k) = ac(k) - 4*x[i]*x[i-k] (for the lags involving i), sign convention.
        # Simpler: try flipping each of a random subset and keep the best.
        # We'll evaluate a small random subset of indices per iteration for speed.
        trial = np.random.choice(n, size=min(n, 64), replace=False)
        best_i = None
        best_s = best
        for i in trial:
            x[i] = -x[i]
            s = _score(x)
            x[i] = -x[i]
            if s < best_s:
                best_s = s
                best_i = i
        if best_i is None:
            # no improving flip in subset; try a full sweep occasionally
            if it % 5 == 0 and _elapsed() < deadline:
                for i in range(n):
                    x[i] = -x[i]
                    s = _score(x)
                    x[i] = -x[i]
                    if s < best_s:
                        best_s = s
                        best_i = i
            if best_i is None:
                break
        x[best_i] = -x[best_i]
        if best_s < best:
            best = best_s
            best_x = x.copy()
    return best_x, best


def _anneal(x, iters, t0, t1, deadline):
    """Simulated annealing over sign flips to escape local minima."""
    n = len(x)
    x = x.astype(np.float64).copy()
    cur = _score(x)
    best = cur
    best_x = x.copy()
    for it in range(iters):
        if _elapsed() > deadline:
            break
        T = t0 * (t1 / t0) ** (it / max(1, iters))
        i = np.random.randint(n)
        x[i] = -x[i]
        new = _score(x)
        if new < cur or np.random.rand() < np.exp(-(new - cur) / max(T, 1e-12)):
            cur = new
            if new < best:
                best = new
                best_x = x.copy()
        else:
            x[i] = -x[i]
    return best_x, best


def _structured_init():
    """Balanced +/-1 initialization with near-zero mean and low initial sidelobes."""
    rng = np.random.default_rng(12345)
    # start balanced
    x = rng.choice([-1.0, 1.0], size=N)
    # force near-zero sum for the normalization to be well-defined and small
    pos = np.where(x > 0)[0]
    neg = np.where(x < 0)[0]
    while len(pos) != len(neg):
        if len(pos) > len(neg):
            i = pos[rng.integers(len(pos))]
            x[i] = -1.0
            pos = np.where(x > 0)[0]
            neg = np.where(x < 0)[0]
        else:
            i = neg[rng.integers(len(neg))]
            x[i] = 1.0
            pos = np.where(x > 0)[0]
            neg = np.where(x < 0)[0]
    return x


def construct():
    deadline = _START + _DEADLINE - 10.0  # 10s safety margin
    rng = np.random.default_rng(0)

    # Build a pool of candidate initializations.
    inits = []
    # 1) structured balanced
    inits.append(_structured_init())
    # 2) a few random balanced inits
    for seed in range(4):
        r = np.random.default_rng(seed + 100)
        x = r.choice([-1.0, 1.0], size=N)
        pos = np.where(x > 0)[0]
        neg = np.where(x < 0)[0]
        while len(pos) != len(neg):
            if len(pos) > len(neg):
                i = pos[r.integers(len(pos))]
                x[i] = -1.0
                pos = np.where(x > 0)[0]
                neg = np.where(x < 0)[0]
            else:
                i = neg[r.integers(len(neg))]
                x[i] = 1.0
                pos = np.where(x > 0)[0]
                neg = np.where(x < 0)[0]
        inits.append(x)

    best_x = None
    best_s = float('inf')

    for x0 in inits:
        if _elapsed() > deadline - 20:
            break
        s0 = _score(x0)
        if s0 < best_s:
            best_s = s0
            best_x = x0.copy()
        # hill-climb from this init
        if _elapsed() < deadline - 10:
            xh, sh = _greedy_flip(x0, max_iters=4000, deadline=deadline - 10, start_score=s0)
            if sh < best_s:
                best_s = sh
                best_x = xh.copy()
        # anneal from the hill-climb result
        if _elapsed() < deadline - 5:
            xa, sa = _anneal(xh, iters=3000, t0=1.0, t1=1e-3, deadline=deadline - 5)
            if sa < best_s:
                best_s = sa
                best_x = xa.copy()
            # quick hill-climb from annealed result
            if _elapsed() < deadline - 2:
                xh2, sh2 = _greedy_flip(xa, max_iters=2000, deadline=deadline - 2, start_score=sa)
                if sh2 < best_s:
                    best_s = sh2
                    best_x = xh2.copy()

    if best_x is None:
        best_x = _structured_init()

    # Final polish: one more greedy pass on the global best if time remains.
    if _elapsed() < deadline - 1:
        xf, sf = _greedy_flip(best_x, max_iters=3000, deadline=deadline - 1, start_score=best_s)
        if sf < best_s:
            best_x = xf
            best_s = sf

    # Return as a sequence of signed step heights (+/-1).
    return best_x
