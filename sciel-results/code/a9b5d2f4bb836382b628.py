"""Rewritten program: direct global construction to minimize the third autocorrelation value.

Strategy (structural change vs. local optimization):
  * The objective 2N*max(conv(h,h))/sum(h)^2 is scale-invariant, so fix sum(h)=1 (positive)
    and minimize 2N*max(corr), where corr[k]=sum_i h[i]h[i+k] is the aperiodic autocorrelation.
    The maximum is attained at lag 0 (= sum h^2), so the goal is to spread the unit mass as
    evenly as possible while keeping the sequence positive (negative spikes raise sum h^2).
  * We therefore (a) evaluate a bank of analytic low-dimensional profiles, (b) build Golay
    complementary pairs (known to flatten the autocorrelation to ~constant, giving sum h^2~N/2),
    and (c) run a randomized global search (simulated annealing + random restarts) with a fast
    FFT-based objective to beat the analytic lower bound.
"""
import time
import numpy as np

N = 400
_DEADLINE = 135.0          # safety margin before the 150 s kill
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _deadline_ok(margin=3.0):
    return _elapsed() < _DEADLINE - margin


def _value(h):
    """Exact objective value (the evaluator's formula)."""
    h = np.asarray(h, dtype=float)
    if h.shape != (N,) or not np.isfinite(h).all():
        return np.inf
    s = float(h.sum())
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    m = float(conv.max())
    return abs(2.0 * N * m / (s * s))


def _fast_value(h):
    """Fast objective via FFT. max of full autocorrelation = max of the rfft-spectrum side."""
    h = np.asarray(h, dtype=float)
    if not np.isfinite(h).all():
        return np.inf
    s = float(h.sum())
    if s == 0.0:
        return np.inf
    size = 2 * N
    f = np.fft.rfft(h, size)
    acf = np.fft.irfft(f * np.conj(f), size)
    m = float(acf[:N].max())
    return abs(2.0 * N * m / (s * s))


def _normalize(h, target=1.0):
    """Scale h so its sum equals target (positive). Returns None if degenerate."""
    s = float(h.sum())
    if not np.isfinite(s) or abs(s) < 1e-12:
        return None
    h = h / s * target
    if not np.isfinite(h).all():
        return None
    return h


# ---------------------------------------------------------------------------
# 1. Analytic low-dimensional profiles (spread the mass, keep it positive)
# ---------------------------------------------------------------------------
def _analytic_profiles():
    i = np.arange(N, dtype=float)
    c = N / 2.0
    t = i / (N - 1.0)
    profs = []

    # constant (the seed) -> value 2.0
    profs.append(np.ones(N))

    # raised-cosine (smooth, positive)
    profs.append(0.5 * (1.0 + np.cos(np.pi * (2.0 * t - 1.0))))

    # Gaussian bumps (several widths), centered
    for w in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]:
        x = (i - c) / (w * N)
        profs.append(np.exp(-0.5 * x * x))

    # sech (flatter-top than Gaussian)
    for w in [0.10, 0.15, 0.20, 0.30, 0.40]:
        x = (i - c) / (w * N)
        profs.append(1.0 / np.cosh(x))

    # tent / triangular
    profs.append(np.clip(1.0 - np.abs(i - c) / c, 0.0, 1.0))

    # plateau + cosine taper at the edges
    for p in [0.4, 0.6, 0.7, 0.8, 0.9]:
        h = np.zeros(N)
        m = int(p * N)
        if m >= 2:
            h[:m] = 1.0
            taper = 0.5 * (1.0 + np.cos(np.pi * (np.arange(N - m) / max(N - m, 1))))
            h[m:] = taper
        profs.append(h)

    # low-order cosine polynomial (positive, smooth)
    for a in [0.1, 0.2, 0.4, 0.6, 0.8]:
        profs.append(1.0 - a * np.cos(2.0 * np.pi * t))

    # quadratic bump
    profs.append(np.clip(1.0 - (2.0 * t - 1.0) ** 2, 0.0, 1.0))

    return profs


# ---------------------------------------------------------------------------
# 2. Golay complementary pairs (flatten the aperiodic autocorrelation)
# ---------------------------------------------------------------------------
def _golay_complementary():
    """Build Golay complementary pairs up to length 2^k (k<=6 -> 64) and tile to N."""
    a = np.array([1.0])
    b = np.array([1.0])
    out = []
    for k in range(1, 7):  # lengths 2,4,8,16,32,64
        la, lb = len(a), len(b)
        a2 = np.concatenate([a, a])
        b2 = np.concatenate([b, -b])
        a = a2
        b = b2
        # complementary: a*a + b*b (aperiodic) is constant outside lag 0
        out.append(a)
        out.append(b)
        # also the sum/difference (single sequence with even flatter max)
        out.append(a + b)
        out.append(a - b)
    # tile each base sequence to length N (circular tiling preserves the flat-ish structure)
    res = []
    for base in out:
        L = len(base)
        reps = int(np.ceil(N / L))
        seq = np.tile(base, reps)[:N]
        res.append(seq)
    return res


# ---------------------------------------------------------------------------
# 3. Randomized global search (simulated annealing + random restarts)
# ---------------------------------------------------------------------------
def _anneal(start, iters, t0, t1, seed):
    rng = np.random.default_rng(seed)
    cur = np.array(start, dtype=float)
    cur = _normalize(cur)
    if cur is None:
        return None
    cur_v = _fast_value(cur)
    best = cur.copy()
    best_v = cur_v
    n = N
    for it in range(iters):
        if _elapsed() > _DEADLINE - 2.0:
            break
        T = t0 * (t1 / t0) ** (it / max(iters, 1))
        # random perturbation: a few Gaussian bumps + small noise
        k = rng.integers(1, 5)
        for _ in range(k):
            c = rng.integers(0, n)
            w = rng.uniform(0.03, 0.25) * n
            amp = rng.uniform(-0.5, 0.5)
            x = (np.arange(n) - c) / w
            cur += amp * np.exp(-0.5 * x * x)
        cur += rng.normal(0.0, 0.05, size=n)
        cur = _normalize(cur)
        if cur is None:
            cur = best.copy()
            continue
        v = _fast_value(cur)
        if v < cur_v or rng.random() < np.exp(-(v - cur_v) / max(T, 1e-9)):
            cur_v = v
            if v < best_v:
                best_v = v
                best = cur.copy()
    return best, best_v


def _global_search(best_h, best_v):
    # Random positive restarts (log-uniform magnitudes)
    rng = np.random.default_rng(7)
    for r in range(60):
        if not _deadline_ok():
            break
        base = rng.uniform(0.5, 2.0, size=N)
        # inject a couple of gentle dips (still mostly positive)
        for _ in range(rng.integers(0, 3)):
            c = rng.integers(0, N)
            w = rng.uniform(0.05, 0.2) * N
            x = (np.arange(N) - c) / w
            base -= rng.uniform(0.1, 0.4) * np.exp(-0.5 * x * x)
        base = np.clip(base, 0.01, None)
        h = _normalize(base)
        if h is None:
            continue
        v = _fast_value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    # Simulated annealing from the current best (and a couple of good restarts)
    starts = [best_h]
    for r in range(4):
        base = rng.uniform(0.5, 2.0, size=N)
        h = _normalize(base)
        if h is not None and _fast_value(h) < best_v:
            starts.append(h)
    for si, st in enumerate(starts):
        if not _deadline_ok():
            break
        res = _anneal(st, iters=1500, t0=0.5, t1=0.01, seed=100 + si)
        if res is not None:
            h, v = res
            if v < best_v:
                best_v = v
                best_h = h.copy()
    return best_h, best_v


def construct():
    global _T0
    _T0 = time.time()

    best_h = np.ones(N)
    best_v = _fast_value(best_h)  # 2.0

    def consider(h):
        nonlocal best_h, best_v
        if h is None:
            return
        h = np.asarray(h, dtype=float)
        if not np.isfinite(h).all():
            return
        s = float(h.sum())
        if abs(s) < 1e-12:
            return
        v = _fast_value(h)
        if v < best_v - 1e-12:
            best_v = v
            best_h = h.copy()

    # (1) analytic profiles
    for p in _analytic_profiles():
        if not _deadline_ok():
            break
        h = _normalize(p)
        consider(h)

    # (2) Golay complementary pairs
    for g in _golay_complementary():
        if not _deadline_ok():
            break
        h = _normalize(g)
        consider(h)
        # also a lightly-smoothed version (reduces tiling discontinuities)
        if _deadline_ok():
            hs = np.convolve(h, np.ones(3) / 3.0, mode='same')
            consider(hs)

    # (3) randomized global search
    best_h, best_v = _global_search(best_h, best_v)

    # Final validation with the exact formula
    s = float(best_h.sum())
    if s == 0.0 or not np.isfinite(best_h).all():
        return np.ones(N)
    v_final = _value(best_h)
    if np.isfinite(v_final) and v_final < 2.0 - 1e-9:
        return best_h
    return np.ones(N)
