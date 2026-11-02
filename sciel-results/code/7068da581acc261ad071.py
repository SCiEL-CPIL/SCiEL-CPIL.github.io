"""Optimized solver for the Erdos minimum-overlap step-height problem."""
import time
import numpy as np

N = 100


def _overlap_at(h, k):
    """Overlap of h with shifted (1-h) at circular shift k (vectorized)."""
    if k == 0:
        return 0.5 - np.dot(h, h)
    a = np.concatenate((h[k:], h[:k]))
    return 0.5 - np.dot(h, a)


def _eval_max(h, ks):
    """Exact maximum overlap over the given shift set."""
    best = 0.0
    for k in ks:
        if k == 0:
            ov = 0.5 - np.dot(h, h)
        else:
            a = np.concatenate((h[k:], h[:k]))
            ov = 0.5 - np.dot(h, a)
        if ov > best:
            best = ov
    return best


def _project(v):
    """Project v onto {0<=x<=1, sum x = N/2} via bisection on the threshold."""
    lo, hi = v.min(), v.max()
    target = 0.5 * N
    for _ in range(80):
        m = 0.5 * (lo + hi)
        s = np.sum(np.clip(v - m, 0.0, 1.0))
        if s > target:
            lo = m
        else:
            hi = m
    return np.clip(v - 0.5 * (lo + hi), 0.0, 1.0)


def _surrogate(h, ks, beta):
    """Log-sum-exp surrogate for max overlap; returns (value, gradient)."""
    vals = np.empty(len(ks))
    for j, k in enumerate(ks):
        if k == 0:
            vals[j] = 0.5 - np.dot(h, h)
        else:
            a = np.concatenate((h[k:], h[:k]))
            vals[j] = 0.5 - np.dot(h, a)
    m = vals.max()
    LSE = m + np.log(np.sum(np.exp(vals - m)))
    w = np.exp(vals - m)
    w = w / w.sum()
    g = np.zeros(N)
    for j, k in enumerate(ks):
        if k == 0:
            g -= 2.0 * h
        else:
            g -= h
            g -= np.roll(h, k)
        g *= w[j]
    return LSE, g


def _refine(h, ks, beta, iters, step):
    """Projected gradient descent on the smooth surrogate."""
    h = h.copy()
    for _ in range(iters):
        _, g = _surrogate(h, ks, beta)
        h = _project(h - step * g)
    return h


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under the 150 s wall-clock limit

    # Focus on the non-trivial shifts; shift 0 gives a constant 0.5.
    ks = list(range(1, N))

    def elapsed():
        return time.time() - start

    # --- Phase 1: coarse search over temperature and step size ---
    best_h = np.full(N, 0.5)
    best_score = _eval_max(best_h, ks)

    for beta in (15.0, 40.0, 80.0, 160.0, 320.0):
        for step in (0.02, 0.05, 0.1, 0.2):
            if elapsed() > deadline - 20.0:
                break
            # random restart
            r = np.random.RandomState(int(beta * 131 + step * 1000) % (2**31))
            x = r.rand(N)
            x = (x - x.min()) / max(x.max() - x.min(), 1e-9)
            x = 0.5 + 0.5 * (x - 0.5)
            h = _refine(x, ks, beta, 400, step)
            s = _eval_max(h, ks)
            if s < best_score:
                best_score = s
                best_h = h.copy()

    # --- Phase 2: high-precision polish on the best candidate ---
    if elapsed() < deadline - 10.0:
        h = best_h.copy()
        # anneal the temperature up for a sharper surrogate
        for beta in (320.0, 640.0, 1280.0, 2560.0, 5120.0):
            if elapsed() > deadline - 10.0:
                break
            h = _refine(h, ks, beta, 300, 0.05)
            s = _eval_max(h, ks)
            if s < best_score:
                best_score = s
                best_h = h.copy()

    # --- Phase 3: exact local solve (epigraph) to tighten the bound ---
    if elapsed() < deadline - 5.0:
        try:
            from scipy.optimize import minimize
            h0 = best_h.copy()

            def fun(x):
                # epigraph: t + small penalty for violations
                t = x[N]
                v = 0.0
                for k in ks:
                    if k == 0:
                        ov = 0.5 - np.dot(x, x)
                    else:
                        a = np.concatenate((x[k:], x[:k]))
                        ov = 0.5 - np.dot(x, a)
                    v += max(0.0, ov - t)
                return t + 1e-3 * v

            def jac(x):
                t = x[N]
                g = np.zeros(N + 1)
                g[N] = 1.0
                for k in ks:
                    if k == 0:
                        ov = 0.5 - np.dot(x, x)
                        if ov > t:
                            g -= 2.0 * x
                            g[N] -= 1.0
                    else:
                        a = np.concatenate((x[k:], x[:k]))
                        ov = 0.5 - np.dot(x, a)
                        if ov > t:
                            g -= x
                            g -= np.roll(x, k)
                            g[N] -= 1.0
                return g

            x0 = np.concatenate((h0, [best_score + 1e-6]))
            res = minimize(fun, x0, jac=jac, method="L-BFGS-B",
                           bounds=[(0.0, 1.0)] * N + [(0.0, 2.0)],
                           options={"maxiter": 400, "ftol": 1e-14, "gtol": 1e-12})
            if elapsed() < deadline - 2.0 and res.x is not None:
                hx = _project(res.x[:N])
                s = _eval_max(hx, ks)
                if s < best_score:
                    best_score = s
                    best_h = hx.copy()
        except Exception:
            pass

    # Final: enforce exact sum and box, return the best heights.
    h = _project(best_h)
    # Nudge to make the sum exactly N/2 within float precision.
    diff = 0.5 * N - h.sum()
    if abs(diff) > 0:
        idx = int(np.argmax(h < 1.0)) if (h < 1.0).any() else 0
        h[idx] += diff
    return h
