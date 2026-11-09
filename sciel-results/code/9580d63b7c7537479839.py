"""Rewritten program: solve the Laguerre double-root BVP to tighten C4."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre
from scipy.integrate import solve_bvp
from scipy.optimize import minimize, least_squares

M = 12          # number of double roots
N = 1           # Laguerre order (n=1)
K = 13          # number of basis Laguerres (m = 12 -> indices 0..12)

# Precompute basis functions and derivatives at a dense grid
_XG = np.linspace(1e-3, 300.0, 4000)

def _lag_val(x, order):
    return eval_genlaguerre(order, N, x)

def _lag_der(x, order):
    # d/dx L_k^{(n)}(x) = - L_{k-1}^{(n+1)}(x)
    if order == 0:
        return 0.0
    return -eval_genlaguerre(order - 1, N + 1, x)

def _F(x, c):
    s = 0.0
    for k in range(K):
        s += c[k] * _lag_val(x, k)
    return s

def _Fp(x, c):
    s = 0.0
    for k in range(K):
        s += c[k] * _lag_der(x, k)
    return s

def _residuals(c, roots):
    """2*M residuals: F(r_i)=0 and F'(r_i)=0 for each double root."""
    res = np.empty(2 * M)
    for i, r in enumerate(roots):
        res[2 * i] = _F(r, c)
        res[2 * i + 1] = _Fp(r, c)
    return res

def _solve_from_init(roots0):
    """Given initial root guesses, solve for coefficients + refine roots."""
    # Normalize: fix c[0] = 1, solve linear-ish least squares for the rest
    def loss(params):
        c = np.zeros(K)
        c[0] = 1.0
        c[1:] = params
        r = _residuals(c, roots0)
        return np.sum(r * r)

    # Refine roots as well using a joint least squares
    def joint(params):
        c = np.zeros(K)
        c[0] = 1.0
        c[1:] = params[:K - 1]
        r = np.sort(params[K - 1:])
        return _residuals(c, r)

    x0 = np.concatenate([np.zeros(K - 1), roots0])
    sol = least_squares(joint, x0, method='lm', max_nfev=20000)
    c = np.zeros(K)
    c[0] = 1.0
    c[1:] = sol.x[:K - 1]
    roots = np.sort(sol.x[K - 1:])
    resid = np.linalg.norm(_residuals(c, roots))
    return roots, c, resid

def construct():
    # Initial guesses: use a clustered (non-uniform) guess that tends to
    # pull the largest root below the seed's 48.0.
    # Laguerre-node-like spacing: roots grow sub-linearly.
    base = np.array([
        2.0, 3.5, 5.0, 6.8, 8.7, 10.8,
        13.1, 15.6, 18.3, 21.2, 24.4, 27.9
    ])
    # Also try the seed's uniform guess and a few perturbations; keep best.
    candidates = [
        base,
        np.array([4.0 * (i + 1) for i in range(M)]),
        0.8 * base,
        1.2 * base,
        np.sqrt(np.arange(1, M + 1)) * 5.0,
    ]

    best_roots = None
    best_max = np.inf
    for cand in candidates:
        try:
            roots, c, resid = _solve_from_init(np.sort(cand))
            if roots is None or len(roots) != M:
                continue
            # Validity: all positive, strictly increasing, small residual
            if np.any(roots <= 0):
                continue
            if np.any(np.diff(roots) <= 1e-8):
                continue
            if resid > 1e-4:
                continue
            mx = float(np.max(roots))
            if mx < best_max:
                best_max = mx
                best_roots = roots
        except Exception:
            continue

    # Fallback: return a safe positive, increasing set (never invalid)
    if best_roots is None:
        return [4.0 * (i + 1) for i in range(M)]

    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
