"""Fresh design: minimize max|autoconv|/(4N) for signed step functions via
continuous L-BFGS-B on a sign-based active set + vectorized peak-targeted polish."""
import time
import numpy as np

N = 400


def _autoconv(z):
    """Full linear autocorrelation of z (symmetric, length 2N-1)."""
    return np.fft.irfft(np.fft.rfft(z, 2 * N - 1) ** 2, 2 * N - 1)


def _maxabs(z):
    ac = _autoconv(z)
    return float(np.max(np.abs(ac)))


def construct():
    t_start = time.time()
    deadline = t_start + 140.0   # safety margin vs 150s wall-clock kill
    rng = np.random.default_rng(12345)

    # --- build a good initial sign vector (greedy: minimize running max) ---
    z = np.empty(N, dtype=np.float64)
    z[0] = 1.0
    # greedy fill: choose sign of next bit to keep autocorr small
    for i in range(1, N):
        # estimate contribution: try +1 and -1, pick the one with smaller peak
        z[i] = 1.0
        # cheap local check via correlation with prefix
        pref = z[:i + 1]
        ac = _autoconv(pref)
        cand1 = float(np.max(np.abs(ac)))
        z[i] = -1.0
        ac = _autoconv(pref)
        cand2 = float(np.max(np.abs(ac)))
        z[i] = 1.0 if cand1 <= cand2 else -1.0

    best_z = z.copy()
    best_val = _maxabs(best_z)

    def objective(s, sign):
        # continuous variable s (real), active set = sign(s)
        zc = sign * np.maximum(s, 0.0)
        # smooth-ish objective: max|ac| approximated by soft-max of squares
        ac = _autoconv(zc)
        a = np.abs(ac)
        # soft max via logsumexp for differentiability
        m = a.max()
        return float(m + np.log(np.sum(np.exp(a - m))) / len(a))

    # --- L-BFGS-B on continuous relaxation, re-signing each eval ---
    from scipy.optimize import minimize
    sign = best_z.copy()
    x0 = np.abs(best_z)  # start at |z|

    def fun(x):
        return objective(x, sign)

    res = None
    try:
        res = minimize(fun, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    except Exception:
        pass

    if res is not None:
        x = np.clip(res.x, -1.0, 1.0)
        new_sign = np.sign(x)
        new_sign[new_sign == 0] = 1.0
        zc = new_sign
        val = _maxabs(zc)
        if val < best_val:
            best_val = val
            best_z = zc.copy()
            sign = new_sign.copy()

    # --- vectorized peak-targeted coordinate polish ---
    z = best_z.copy()
    iters = 0
    while time.time() < deadline - 2.0:
        ac = _autoconv(z)
        a = np.abs(ac)
        peak_lag = int(np.argmax(a))
        # coordinate descent: flip bits that reduce peak
        improved = False
        # candidate positions to try flipping (spread across array)
        idxs = np.arange(N)
        # evaluate flipping each: ac change for flipping bit j is -2*z[j]*(z * z_j_shift)
        # do a batched cheap update: for each j, delta_ac = -2*z[j]*corr where corr[l] = sum_k z[k]*z[k + (lag - j)]
        # Use FFT-based per-flip? Instead do a limited random subset for speed.
        step = 8
        for j in range(0, N, step):
            # compute new ac after flipping j
            z[j] = -z[j]
            ac2 = _autoconv(z)
            a2 = np.max(np.abs(ac2))
            if a2 < best_val - 1e-9:
                best_val = a2
                best_z = z.copy()
                improved = True
            else:
                z[j] = -z[j]  # revert
        iters += 1
        if not improved and iters > 20:
            break

    # final polish: full sweep until no improvement
    z = best_z.copy()
    while time.time() < deadline - 1.0:
        any_improve = False
        for j in range(N):
            z[j] = -z[j]
            v = _maxabs(z)
            if v < best_val - 1e-9:
                best_val = v
                best_z = z.copy()
                any_improve = True
            else:
                z[j] = -z[j]
        if not any_improve:
            break

    # convert sign vector back to signed step heights in {+1, -1}
    return best_z
