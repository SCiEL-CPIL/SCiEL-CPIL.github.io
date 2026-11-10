# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

_DEADLINE = time.time() + 580.0  # safety margin before 630s kill

# --- Precompute real orthonormal degree-10 harmonics at quadrature points ---
_k = 10
_ntheta = 70
_nphi = 130

# Gauss-Legendre in cos(theta); weight = sin(theta)
_x, _w = np.polynomial.legendre.leggauss(_ntheta)
_cos = _x
_sin = np.sqrt(np.maximum(0.0, 1.0 - _x * _x))
_wt = _w * _sin  # surface-measure weight (dOmega = sin(theta) dtheta dphi)
# phi equally spaced; total weight over all points = 4*pi
_phi = np.linspace(0.0, 2.0 * np.pi, _nphi, endpoint=False)
_wphi = 2.0 * np.pi / _nphi
_wfull = np.repeat(_wt, _nphi)  # length ntheta*nphi

# cos/sin tables
_cphi = np.cos(_phi)
_sphi = np.sin(_phi)

# Associated Legendre P_10^m with Condon-Shortley phase
def _plm(m, x):
    return lpmv(m, _k, x)  # includes (-1)^m

# Build M: shape (21, ntheta*nphi), row i -> m = i-10
_mvals = np.arange(-_k, _k + 1)
_M = np.empty((2 * _k + 1, _ntheta * _nphi), dtype=np.float64)
for idx, m in enumerate(_mvals):
    am = abs(m)
    Nm = np.sqrt(21.0 / (4.0 * np.pi) * float(np.math.factorial(_k - am)) /
                 float(np.math.factorial(_k + am)))
    P = _plm(am, _cos)  # length ntheta
    if m == 0:
        Y = Nm * P  # ntheta
    elif m > 0:
        Y = np.sqrt(2.0) * Nm * P * _cphi[m]  # ntheta x nphi
    else:
        Y = np.sqrt(2.0) * Nm * P * _sphi[am]  # ntheta x nphi
    # tile to ntheta*nphi
    if m == 0:
        _M[idx, :] = np.repeat(Y, _nphi)
    else:
        _M[idx, :] = np.tile(Y, _ntheta)

# Orthonormality sanity: M W M^T ~ I
# (skip for speed; trust construction)

# Quadrature normalization: mean f^4 = (1/4pi) sum wfull f^4
_Wsum = _wfull.sum()  # ~ 4*pi


def _ratio(c):
    """R = ||f||_4/||f||_2 with f normalized to unit L2 (== (mean f^4)^{1/4})."""
    f = _M.dot(c)  # ntheta*nphi
    l2sq = _wfull.dot(f * f)
    if l2sq <= 0.0:
        return 1.0
    l4 = _wfull.dot(f * f * f * f)
    # R = (l4/4pi)^(1/4) / (l2sq/4pi)^(1/2) ; with l2sq=4pi -> R=(l4/4pi)^(1/4)
    R = (l4 / _Wsum) ** 0.25 / (l2sq / _Wsum) ** 0.5
    return R


def _objective(c):
    """Flatness objective: (mean f^4 - 1/5)^2, f unit-L2. Min at flat limit."""
    f = _M.dot(c)
    l2sq = _wfull.dot(f * f)
    if l2sq <= 0.0:
        return 1.0
    l4 = _wfull.dot(f * f * f * f)
    mean4 = l4 / _Wsum
    # f normalized to unit L2 -> mean f^4 = (l4/4pi)/(l2sq/4pi)^2 = l4*4pi/l2sq^2
    mean4n = l4 * _Wsum / (l2sq * l2sq)
    d = mean4n - 0.2
    return d * d


def _flatness_score(c):
    """The reported value = 1/R (higher better)."""
    return 1.0 / _ratio(c)


def _optimize_from(x0, iters=4000):
    res = minimize(_objective, x0, method='L-BFGS-B',
                   options={'maxiter': iters, 'maxls': 50,
                            'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 20000})
    return res.x


def _diverse_inits():
    inits = []
    # single modes
    for i in range(2 * _k + 1):
        v = np.zeros(2 * _k + 1)
        v[i] = 1.0
        inits.append(v)
    # structured: cos^m(phi) dominant (m even)
    for m in range(0, 2 * _k + 1, 2):
        v = np.zeros(2 * _k + 1)
        v[m - _k] = 1.0
        inits.append(v)
    # random
    rng = np.random.default_rng(0)
    for _ in range(20):
        inits.append(rng.standard_normal(2 * _k + 1))
    # random sparse (few nonzero)
    for _ in range(15):
        v = np.zeros(2 * _k + 1)
        idx = rng.choice(2 * _k + 1, size=3, replace=False)
        v[idx] = rng.standard_normal(3)
        inits.append(v)
    return inits


def rudin_problem10() -> np.ndarray:
    global _DEADLINE
    best = None
    best_score = -np.inf

    def _consider(c):
        nonlocal best, best_score
        if c is None:
            return
        c = np.asarray(c, dtype=np.float64)
        if c.size != 2 * _k + 1 or not np.all(np.isfinite(c)):
            return
        s = _flatness_score(c)
        if s > best_score:
            best_score = s
            best = c.copy()

    inits = _diverse_inits()
    for x0 in inits:
        if time.time() > _DEADLINE:
            break
        try:
            xopt = _optimize_from(x0, iters=3000)
            _consider(xopt)
        except Exception:
            pass

    # Random restarts with the best as a warm base
    rng = np.random.default_rng(12345)
    base = best if best is not None else np.ones(2 * _k + 1)
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        if best is not None:
            x0 = base + 0.5 * rng.standard_normal(2 * _k + 1)
        else:
            x0 = rng.standard_normal(2 * _k + 1)
        try:
            xopt = _optimize_from(x0, iters=4000)
            _consider(xopt)
        except Exception:
            pass

    # Final polish from best with tight tolerances
    if best is not None and time.time() < _DEADLINE:
        try:
            xopt = _optimize_from(best, iters=8000)
            _consider(xopt)
        except Exception:
            pass

    if best is None:
        best = np.ones(2 * _k + 1, dtype=np.float64)

    # Validate and scale into safe range
    best = np.asarray(best, dtype=np.float64)
    if best.size != 2 * _k + 1 or not np.all(np.isfinite(best)):
        best = np.ones(2 * _k + 1, dtype=np.float64)
    mx = np.max(np.abs(best))
    if mx < 1e-3:
        best = best / mx * 1.0
    elif mx > 1e5:
        best = best / mx * 1.0
    return best


# EVOLVE-BLOCK-END
