# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

_DEADLINE = 630.0
_start = time.time()

def _remaining():
    return _DEADLINE - 30.0 - time.time()  # 30s safety margin

def _build_basis():
    """Return (M2, A4) discrete moment matrices and the quadrature point count."""
    k = 10
    # Gauss-Legendre in x = cos theta
    x, w = np.polynomial.legendre.leggauss(70)
    # phi grid
    nphi = 130
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    dphi = 2.0 * np.pi / nphi
    # weights: w_leg[i] * dphi
    W = np.outer(w, np.full(nphi, dphi))  # shape (70,130)
    # Build basis values at all points
    # m from -10..10
    ms = np.arange(-k, k + 1)
    abs_ms = np.abs(ms)
    N = np.sqrt(21.0 / (4.0 * np.pi) * (factorial(k - abs_ms) / factorial(k + abs_ms)))
    # P_10^m(x) with Condon-Shortley phase
    P = np.empty((k + 1, len(x)))  # P[m, i]
    for m in range(k + 1):
        P[m, :] = lpmv(m, k, x)
    # basis: shape (21, 70, 130)
    B = np.empty((2 * k + 1, len(x), nphi))
    for idx, m in enumerate(ms):
        am = abs(m)
        if m == 0:
            B[idx, :, :] = N[0] * P[am, :][:, None]
        elif m > 0:
            B[idx, :, :] = np.sqrt(2.0) * N[am] * P[am, :][:, None] * np.cos(m * phi)[None, :]
        else:
            B[idx, :, :] = np.sqrt(2.0) * N[am] * P[am, :][:, None] * np.sin(am * phi)[None, :]
    # Flatten to (21, npts)
    Bf = B.reshape(2 * k + 1, -1)
    Wf = W.reshape(-1)
    # M2_{mn} = sum_p Wf[p] Bf[m,p] Bf[n,p]
    M2 = (Bf * Wf[None, :]).dot(Bf.T)
    # A4_{m1,m2,m3,m4} = sum_p Wf[p] Bf[m1,p] Bf[m2,p] Bf[m3,p] Bf[m4,p]
    # Compute via einsum on (21, npts)
    Wsqrt = np.sqrt(np.maximum(Wf, 0.0))
    Bw = Bf * Wsqrt[None, :]
    A4 = np.einsum('mp,np,qp,rp->mnpr', Bw, Bw, Bw, Bw)
    return M2, A4


def _objective_and_grad(c, M2, A4):
    c = np.asarray(c, dtype=np.float64)
    # Q2 = c^T M2 c
    M2c = M2.dot(c)
    Q2 = float(c.dot(M2c))
    if Q2 <= 0.0:
        return 1e30, np.zeros_like(c)
    # Q4 = sum_{m1..m4} A4[m1,m2,m3,m4] c1 c2 c3 c4
    # Use einsum
    Q4 = float(np.einsum('ijkl,i,j,k,l->', A4, c, c, c, c))
    if Q4 <= 0.0:
        return 1e30, np.zeros_like(c)
    g = 0.25 * np.log(Q4) - 0.5 * np.log(Q2)
    # grad
    # dQ4/dc_i = 4 * sum_{jkl} A4[i,j,k,l] c_j c_k c_l  (symmetric in all 4)
    # Since A4 is symmetric under permutation, dQ4/dc_i = 4 * einsum('ijkl,j,k,l->i', A4, c,c,c)
    dQ4 = 4.0 * np.einsum('ijkl,j,k,l->i', A4, c, c, c)
    grad = dQ4 / Q4 - M2c / Q2
    return g, grad


def rudin_problem10() -> np.ndarray:
    k = 10
    n = 2 * k + 1
    if _remaining() < 5.0:
        return np.ones(n, dtype=np.float64)
    M2, A4 = _build_basis()
    if _remaining() < 5.0:
        return np.ones(n, dtype=np.float64)

    best_c = np.ones(n, dtype=np.float64)
    best_val = np.inf

    # candidate starts
    starts = [np.ones(n, dtype=np.float64)]
    # random starts
    rng = np.random.default_rng(12345)
    for _ in range(6):
        s = rng.standard_normal(n)
        s /= np.linalg.norm(s)
        starts.append(s)
    # a few structured starts: emphasize low |m| or high |m|
    for _ in range(4):
        s = np.ones(n)
        s[rng.choice(n, size=8, replace=False)] = 0.0
        s /= np.linalg.norm(s)
        starts.append(s)

    def _eval(c):
        nonlocal best_c, best_val
        g, _ = _objective_and_grad(c, M2, A4)
        if np.isfinite(g) and g < best_val:
            best_val = g
            best_c = c.copy()
        return g

    for s in starts:
        if _remaining() < 15.0:
            break
        s = np.asarray(s, dtype=np.float64)
        norm = np.linalg.norm(s)
        if norm > 0:
            s = s / norm
        try:
            res = minimize(_objective_and_grad, s, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 400, 'gtol': 1e-10, 'ftol': 1e-14})
            if res.success or np.isfinite(res.fun):
                _eval(res.x)
        except Exception:
            pass

    # final polish with tighter tol if time
    if _remaining() > 30.0:
        try:
            res = minimize(_objective_and_grad, best_c, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 600, 'gtol': 1e-12, 'ftol': 1e-16})
            if np.isfinite(res.fun):
                _eval(res.x)
        except Exception:
            pass

    # validate
    out = np.asarray(best_c, dtype=np.float64)
    if out.shape != (n,) or not np.all(np.isfinite(out)):
        out = np.ones(n, dtype=np.float64)
    mx = np.max(np.abs(out))
    if mx < 1e-3 or mx > 1e5:
        if mx > 0:
            out = out / mx * 1.0
        else:
            out = np.ones(n, dtype=np.float64)
    return out


# EVOLVE-BLOCK-END
