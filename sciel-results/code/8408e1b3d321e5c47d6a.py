"""Fresh design: smooth log-sum-exp surrogate + SLSQP, diverse starts, deadline guard, exact polish."""
import time
import numpy as np

N = 100
T0 = time.time()
DEADLINE = 150.0
MARGIN = 10.0  # safety margin before wall-clock kill


def _corr_values(h):
    """All full-correlation values of h with (1-h)."""
    return np.correlate(h, 1.0 - h, mode="full")


def _value(h):
    """The evaluator's raw objective (lower is better)."""
    return float(np.max(_corr_values(h)) / N * 2.0)


def _surrogate(h, tau):
    """Smooth log-sum-exp surrogate of max(corr)/N * 2."""
    c = _corr_values(h)
    m = np.max(c)
    # log-sum-exp of (c - m) for stability
    s = np.log(np.sum(np.exp(tau * (c - m)))) + tau * m
    return s / N * 2.0


def _grad_surrogate(h, tau):
    """Gradient of the surrogate w.r.t. h."""
    c = _corr_values(h)
    m = np.max(c)
    e = np.exp(tau * (c - m))
    w = e / np.sum(e)
    # d c_k / d h  (correlate(h, 1-h, full)[k] = sum_i h[i]*(1-h)[i + k - N + 1])
    # derivative w.r.t. h[i]: sum_k w_k * d c_k/d h[i]
    # c_k = sum_i h[i]*(1-h)[i+k-N+1]
    # dc_k/dh[i] = (1-h)[i+k-N+1] - h[i]*delta(i+k-N+1 == i)
    #            = (1-h)[i+k-N+1] - h[i]*delta(k == N-1)
    # We build gradient via correlation of w with (1-h) and w with h.
    g1 = np.correlate(w, 1.0 - h, mode="full")  # g1[i] = sum_k w[k]*(1-h)[i+k-(N-1)]
    g2 = np.correlate(w, h, mode="full")        # g2[i] = sum_k w[k]*h[i+k-(N-1)]
    grad = g1 - g2 * w[N - 1]
    return grad / N * 2.0


def _make_constraints():
    cons = []
    cons.append({
        "type": "eq",
        "fun": lambda h: np.sum(h) - N / 2.0,
        "jac": lambda h: np.ones(N),
    })
    return cons


def _bounds():
    return [(0.0, 1.0) for _ in range(N)]


def _slsqp(h0, tau, cons, bounds, maxiter, ftol):
    from scipy.optimize import minimize
    res = minimize(
        _surrogate,
        h0,
        args=(tau,),
        jac=lambda h, t: _grad_surrogate(h, t),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
    )
    return res.x


def _repair(h):
    """Clip to [0,1] and rescale to satisfy sum = N/2 within tolerance."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    s = np.sum(h)
    target = N / 2.0
    if s > 0:
        h = h * (target / s)
    # After scaling may exceed 1; clip and fix sum with a few iterations.
    for _ in range(50):
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        if abs(s - target) < 1e-12:
            break
        # add/subtract on the free (non-saturated) entries
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += (target - s) / np.sum(free)
        else:
            break
    h = np.clip(h, 0.0, 1.0)
    return h


def _valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1.0 + 1e-12):
        return False
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return False
    return True


def construct():
    global T0
    cons = _make_constraints()
    bounds = _bounds()

    # Diverse initial points.
    rng = np.random.default_rng(0)
    starts = []
    # 1) alternating high/low
    alt = np.tile([0.9, 0.1], N // 2)
    starts.append(alt.copy())
    # 2) block pattern
    blk = np.zeros(N)
    blk[::2] = 0.9
    blk[1::2] = 0.1
    starts.append(blk.copy())
    # 3) cosine wave
    starts.append(0.5 + 0.45 * np.cos(2 * np.pi * 2 * np.arange(N) / N))
    # 4) a few random
    for _ in range(4):
        starts.append(0.5 + 0.45 * rng.standard_normal(N))
    starts = [np.clip(s, 0.0, 1.0) for s in starts]
    starts = [_repair(s) for s in starts]

    best_h = None
    best_val = np.inf
    seed_h = np.full(N, 0.5)
    best_h = seed_h.copy()
    best_val = _value(seed_h)

    # Coarse-to-fine tau for the surrogate.
    tau_schedule = [20.0, 5.0, 1.5, 0.5, 0.15]

    for s in starts:
        h = s.copy()
        for tau in tau_schedule:
            if time.time() > DEADLINE - MARGIN:
                break
            try:
                h2 = _slsqp(h, tau, cons, bounds, maxiter=80, ftol=1e-9)
                h2 = _repair(h2)
                if _valid(h2):
                    v = _value(h2)
                    if v < best_val:
                        best_val = v
                        best_h = h2.copy()
                h = h2
            except Exception:
                pass
        if time.time() > DEADLINE - MARGIN:
            break

    # Final high-precision polish on the best point with a very small tau.
    if time.time() < DEADLINE - MARGIN:
        try:
            hp = _slsqp(best_h, 0.05, cons, bounds, maxiter=200, ftol=1e-12)
            hp = _repair(hp)
            if _valid(hp):
                vp = _value(hp)
                if vp < best_val:
                    best_val = vp
                    best_h = hp.copy()
        except Exception:
            pass

    # Ensure the returned point is valid; fall back to seed if not.
    if not _valid(best_h):
        best_h = seed_h.copy()

    return best_h
