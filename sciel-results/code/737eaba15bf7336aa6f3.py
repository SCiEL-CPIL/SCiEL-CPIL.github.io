"""Second autocorrelation inequality: maximize l2/(norm1*norminf) over 50 step heights.

Key ideas:
  * The objective is scale-invariant (only the *shape* of h matters), so we
    normalise the candidate to unit L1 before scoring.
  * We explore a rich pool of analytic shapes (Gaussian, super-Gaussian,
    asymmetric, trapezoid, tent, cosine, bimodal, power-law, plateau) plus a
    few random smooth profiles.
  * We refine each promising candidate two ways:
      1) full 50-dim Nelder-Mead on the log-parametrised heights
         h_i = exp(u_i)  (enforces non-negativity, avoids boundary issues)
      2) low-dimensional DCT-coefficient optimisation:
         h = |IDCT(c)|  with c restricted to the first ~8 coefficients,
         letting a gradient-based L-BFGS-B explore global shape features
         (symmetry, smoothness, low-frequency content) without the local
         minima traps of explicit basis-function combinations.
  * A final L-BFGS-B polish on the full 50-dim (bounded) space tightens the
    result.
  * An explicit wall-clock deadline guard returns the best result found with a
    safety margin before the 150 s kill.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

N = 50
DEADLINE = 120.0          # safety margin below the 150 s wall-clock kill
_start = time.time()


def _time_left():
    return DEADLINE - (time.time() - _start)


def _value_from_g(g):
    """Fast objective from the autocorrelation g (length 2N-1 = 99)."""
    w = 1.0 / (N + 1)
    g2 = float(np.dot(g, g))
    gc = float(np.dot(g[:-1], g[1:]))
    # l2 = (w/3) * (g[0]^2 + 2*sum(g^2) + 2*sum(g[:-1]*g[1:]))
    #     = (w/3) * (3*g2 + 2*gc)
    l2 = (w / 3.0) * (3.0 * g2 + 2.0 * gc)
    norm1 = float(np.sum(np.abs(g))) / (N + 1)
    norminf = float(np.max(np.abs(g)))
    denom = norm1 * norminf
    if denom <= 0.0:
        return 0.0
    return l2 / denom


def _value(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)) or np.max(h) <= 0.0:
        return 0.0
    g = np.convolve(h, h)
    return _value_from_g(g)


def _neg_value(h):
    return -_value(h)


# ---------------------------------------------------------------------------
# Shape generators (each returns a nonnegative length-N array)
# ---------------------------------------------------------------------------
def _gaussian(a, center, width):
    x = np.arange(N)
    return a * np.exp(-a * (x - center) ** 2 / (2.0 * width * width))


def _super_gaussian(a, center, width, p):
    x = (np.arange(N) - center) / width
    return a * np.exp(-np.abs(x) ** p)


def _asymmetric_gaussian(a, center, left_w, right_w):
    x = np.arange(N) - center
    h = np.zeros(N)
    h[x < 0] = a * np.exp(-x[x < 0] ** 2 / (2.0 * left_w * left_w))
    h[x >= 0] = a * np.exp(-x[x >= 0] ** 2 / (2.0 * right_w * right_w))
    return h


def _trapezoid(a, inner, outer):
    c = (N - 1) / 2.0
    x = np.abs(np.arange(N) - c)
    h = np.zeros(N)
    mask_inner = x <= inner
    mask_outer = (x > inner) & (x <= outer)
    h[mask_inner] = a
    h[mask_outer] = a * (1.0 - (x[mask_outer] - inner) / (outer - inner))
    return h


def _tent(a, halfw):
    c = (N - 1) / 2.0
    return a * np.exp(-np.abs(np.arange(N) - c) / halfw)


def _cosine(a, center, width):
    x = (np.arange(N) - center) / width
    return a * (0.5 + 0.5 * np.cos(np.pi * np.clip(x, -1.0, 1.0)))


def _bimodal(a, sep, width):
    c = (N - 1) / 2.0
    x = np.arange(N)
    return a * (np.exp(-((x - (c - sep)) ** 2) / (2.0 * width * width)) +
                np.exp(-((x - (c + sep)) ** 2) / (2.0 * width * width)))


def _powerlaw(a, beta):
    c = (N - 1) / 2.0
    return a * np.exp(-beta * np.abs(np.arange(N) - c) ** 0.7)


def _plateau(a, halfw):
    c = (N - 1) / 2.0
    return a * np.exp(-((np.arange(N) - c) / halfw) ** 2)


def _random_smooth(rng, n=8):
    """Random smooth nonnegative profile via random DCT coefficients."""
    c = rng.normal(size=n) * np.exp(-np.arange(n) * 0.3)
    c[0] = abs(c[0]) + 1.0  # ensure positive DC
    h = np.abs(idct(c, type=2, n=N, norm='ortho'))
    return h


# ---------------------------------------------------------------------------
# DCT-coefficient low-dimensional optimisation
# ---------------------------------------------------------------------------
def _dct_to_h(coeffs):
    """Map a DCT coefficient vector to a nonnegative length-N array.

    h = |idct(coeffs)|, with a small floor to keep strictly positive.
    """
    h = np.abs(idct(coeffs, type=2, n=N, norm='ortho'))
    return h + 1e-12


def _neg_value_dct(coeffs):
    h = _dct_to_h(coeffs)
    return -_value(h)


def _refine_dct(h0, ncoeff=8, maxiter=200):
    """Optimise the first ncoeff DCT coefficients to improve h0."""
    c0 = dct(h0, type=2, norm='ortho')[:ncoeff]
    try:
        res = minimize(_neg_value_dct, c0.copy(), method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxls': 30})
        if np.all(np.isfinite(res.x)):
            h = _dct_to_h(res.x)
            return h
    except Exception:
        pass
    return h0


# ---------------------------------------------------------------------------
# Full 50-dim refinement (log-parametrised Nelder-Mead + bounded L-BFGS-B)
# ---------------------------------------------------------------------------
def _neg_value_log(u):
    h = np.exp(u)
    return -_value(h)


def _refine_full(h0, label=""):
    """Full 50-dim local refinement of a candidate shape."""
    h0 = np.asarray(h0, dtype=float)
    h0 = np.maximum(h0, 1e-12)
    best = h0.copy()
    bestv = _value(h0)

    # 1) Nelder-Mead on log-parameters (robust, non-negativity guaranteed)
    if _time_left() > 3.0:
        try:
            u0 = np.log(h0)
            res = minimize(_neg_value_log, u0.copy(), method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-8,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)):
                cand = np.exp(res.x)
                v = _value(cand)
                if v > bestv:
                    bestv, best = v, cand
        except Exception:
            pass

    # 2) L-BFGS-B polish (bounded, smooth)
    if _time_left() > 3.0:
        try:
            res = minimize(_neg_value, best.copy(), method='L-BFGS-B',
                           bounds=[(0.0, None)] * N,
                           options={'maxiter': 200, 'maxls': 20})
            if np.all(np.isfinite(res.x)):
                cand = np.maximum(res.x, 0.0)
                v = _value(cand)
                if v > bestv:
                    bestv, best = v, cand
        except Exception:
            pass

    return best, bestv


def _refine_dct_then_full(h0, ncoeff=8):
    """DCT low-dim optimisation followed by full 50-dim polish."""
    h_dct = _refine_dct(h0, ncoeff=ncoeff)
    return _refine_full(h_dct, label="dct")


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------
def _build_candidates(rng):
    c = (N - 1) / 2.0
    candidates = []

    # Uniform seed
    candidates.append(np.ones(N))

    # Gaussian family
    for a in (0.5, 1.0, 2.0, 4.0, 8.0):
        for center in (c, c - 4.0, c + 4.0):
            for width in (1.5, 2.5, 4.0, 6.0, 9.0, 14.0):
                candidates.append(_gaussian(a, center, width))

    # Super-Gaussian family (sharper peaks)
    for a in (1.0, 2.0, 4.0):
        for width in (2.0, 4.0, 7.0):
            for p in (3.0, 4.0, 6.0):
                candidates.append(_super_gaussian(a, c, width, p))

    # Asymmetric Gaussian family
    for a in (1.0, 2.0):
        for lw in (2.0, 4.0, 7.0):
            for rw in (2.0, 4.0, 7.0):
                if lw != rw:
                    candidates.append(_asymmetric_gaussian(a, c, lw, rw))

    # Trapezoid family
    for a in (1.0, 2.0):
        for inner in (2.0, 4.0, 6.0):
            for outer in (8.0, 12.0, 16.0, 22.0):
                if outer > inner + 1:
                    candidates.append(_trapezoid(a, inner, outer))

    # Tent family
    for a in (1.0, 2.0):
        for halfw in (2.0, 4.0, 6.0, 9.0, 12.0):
            candidates.append(_tent(a, halfw))

    # Cosine family
    for a in (1.0, 2.0):
        for width in (4.0, 7.0, 10.0, 14.0):
            candidates.append(_cosine(a, c, width))

    # Bimodal family
    for a in (1.0, 2.0):
        for sep in (3.0, 5.0, 7.0):
            for width in (2.0, 3.5, 5.0):
                candidates.append(_bimodal(a, sep, width))

    # Power-law family
    for a in (1.0, 2.0):
        for beta in (0.3, 0.6, 1.0, 1.6):
            candidates.append(_powerlaw(a, beta))

    # Plateau family
    for a in (1.0, 2.0):
        for halfw in (2.0, 4.0, 6.0, 8.0, 11.0):
            candidates.append(_plateau(a, halfw))

    # Random smooth profiles
    for _ in range(30):
        candidates.append(_random_smooth(rng))

    return candidates


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct():
    global _start
    _start = time.time()

    rng = np.random.default_rng(42)
    candidates = _build_candidates(rng)

    # Phase 1: cheap screening over the whole candidate pool
    best = np.ones(N)
    bestv = _value(best)
    scored = []
    for h in candidates:
        if _time_left() < 3.0:
            break
        h = np.maximum(h, 0.0)
        if not np.all(np.isfinite(h)) or np.max(h) <= 0.0:
            continue
        v = _value(h)
        scored.append((v, h))
        if v > bestv:
            bestv, best = v, h

    # Phase 2: refine the top shapes
    scored.sort(key=lambda t: -t[0])
    top_k = min(15, len(scored))
    for v, h in scored[:top_k]:
        if _time_left() < 5.0:
            break
        # DCT low-dim + full polish
        rh, rv = _refine_dct_then_full(h, ncoeff=8)
        if rv > bestv:
            bestv, best = rv, rh

    # Phase 3: refine the uniform seed (safe baseline)
    if _time_left() > 5.0:
        rh, rv = _refine_full(np.ones(N), label="seed")
        if rv > bestv:
            bestv, best = rv, rh

    # Phase 4: a few extra random restarts if time permits
    for _ in range(5):
        if _time_left() < 8.0:
            break
        h_rand = _random_smooth(rng, n=10)
        rh, rv = _refine_dct_then_full(h_rand, ncoeff=10)
        if rv > bestv:
            bestv, best = rv, rh

    # Final safety: ensure a valid, finite, nonnegative array is returned.
    best = np.asarray(best, dtype=float)
    if not (np.all(np.isfinite(best)) and np.all(best >= 0.0) and np.max(best) > 0.0):
        best = np.ones(N)
    return [float(x) for x in best]


if __name__ == "__main__":
    h = construct()
    print("value:", _value(h), "n:", len(h), "min:", min(h))
