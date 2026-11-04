"""Signed equal-width step functions minimizing max|autoconv| / (integral)^2.

Design: the objective is scale-invariant (both numerator and denominator are
quadratic in f), so we fix ||f||_2 = 1 and minimize the peak of the linear
autocorrelation, then evaluate the true ratio.  Projected gradient descent on
the unit sphere with periodic rounding to +/-1.
"""
import time
import numpy as np

N = 400
DEADLINE = 150.0
SAFETY = 8.0
RNG_SEED = 20240517


def _autocorr_lags(f):
    """Linear autocorrelation g_k = sum_i f_i f_{i+k} for k = -(N-1)..(N-1)."""
    n = len(f)
    M = 1
    while M < 2 * n:
        M *= 2
    F = np.fft.rfft(f, M)
    P = F * np.conj(F)
    full = np.fft.irfft(P, M)
    return full[:2 * n - 1]


def _autocorr_grads(f, g):
    """Gradients d g_k / d f_j for all lags k and positions j (shape (2N-1, N))."""
    n = len(f)
    G = np.fft.rfft(g, 2 * n)
    f_pad = np.concatenate([f, np.zeros(n)])
    Fp = np.fft.rfft(f_pad, 2 * n)
    C = np.fft.irfft(G * Fp, 2 * n)
    return C[:, :n]


def _score(f):
    g = _autocorr_lags(f)
    S = f.sum()
    if abs(S) < 1e-30:
        return float("inf")
    return float(np.max(np.abs(g)) / (S * S))


def _grad_maxabs(g):
    """Gradient of max_k |g_k| w.r.t. g: one-hot at the active lag(s), signed."""
    idx = np.argmax(np.abs(g))
    sign = 1.0 if g[idx] >= 0 else -1.0
    grad = np.zeros_like(g)
    grad[idx] = sign
    return grad


def _project_unit(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return None
    return v / n


def _round(f):
    r = np.sign(f)
    r[r == 0] = 1.0
    return r


def construct():
    t0 = time.time()
    best_f = np.ones(N, dtype=np.float64)
    best_score = _score(best_f)

    def time_ok():
        return (time.time() - t0) < (DEADLINE - SAFETY)

    # Start from several sign patterns.
    starts = [np.ones(N, dtype=np.float64)]
    rng = np.random.default_rng(RNG_SEED)
    starts.append(_project_unit(rng.standard_normal(N)))
    starts.append(_project_unit(rng.standard_normal(N)))
    # A few low-frequency sign structures (alternating blocks).
    for bl in (2, 4, 8, 16, 32):
        s = np.sign(np.sin(np.arange(N) * (2.0 * np.pi / bl)))
        s[s == 0] = 1.0
        starts.append(_project_unit(s.astype(np.float64)))

    for s0 in starts:
        f = s0.copy()
        f = _project_unit(f)
        if f is None:
            continue
        sc = _score(f)
        if sc < best_score:
            best_score, best_f = sc, f.copy()
        if not time_ok():
            break

        lr = 0.15
        for step in range(300000):
            if not time_ok():
                break
            g = _autocorr_lags(f)
            grad_g = _grad_maxabs(g)
            # df = grad_g @ Gmat  (Gmat = d g_k / d f_j)
            Gmat = _autocorr_grads(f, grad_g)
            df = Gmat @ f
            f = f - lr * df
            f = _project_unit(f)
            if f is None:
                break
            # Periodic rounding + occasional restart of step size.
            if step % 250 == 0:
                fr = _round(f)
                sc_r = _score(fr)
                if sc_r < best_score:
                    best_score, best_f = sc_r, fr.copy()
                # Snap to rounded if it is better on the sphere objective.
                gr = _autocorr_lags(fr)
                if np.max(np.abs(gr)) < np.max(np.abs(g)) * 1.02:
                    f = _project_unit(fr)
            if step % 4000 == 0:
                sc = _score(f)
                if sc < best_score:
                    best_score, best_f = sc, f.copy()
        # Finalize this start.
        fr = _round(f)
        sc_r = _score(fr)
        if sc_r < best_score:
            best_score, best_f = sc_r, fr.copy()

    # Final polish on the best: a short round/gradient refinement.
    f = best_f.copy()
    for _ in range(2000):
        if not time_ok():
            break
        fr = _round(f)
        sc_r = _score(fr)
        if sc_r < best_score:
            best_score, best_f = sc_r, fr.copy()
        g = _autocorr_lags(f)
        grad_g = _grad_maxabs(g)
        Gmat = _autocorr_grads(f, grad_g)
        df = Gmat @ f
        f = f - 0.05 * df
        f = _project_unit(f)
        if f is None:
            break

    # Return the best signed step heights found.
    return np.asarray(best_f)


if __name__ == "__main__":
    res = construct()
    print("score:", _score(res), "sum:", res.sum(), "norm:", np.linalg.norm(res))
